//! `llamacpp-launcher launch pi --model flint`
//!
//! Installs llama.cpp under `~/.llamacpp-launcher/llama` and Pi into
//! `~/.local/bin` if they are missing, starts the router, and execs Pi.
//! `plan`, `install`, and `uninstall` are the same binary.

use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};
use std::time::Duration;

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use installer::cli::{self, arg::Instrumentation, CommandExecute};
use installer::planner::Planner;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::process::{Child, Command};

use llamacpp::common::{self, DEFAULT_HOST, DEFAULT_PORT, PI_PROVIDER};
use llamacpp::gpu::Machine;
use llamacpp::planner::Router;

const APP: cli::App = cli::App {
    product: "llamacpp-launcher",
    binary_name: "llamacpp-launcher",
    version: env!("CARGO_PKG_VERSION"),
    env_prefix: "LLAMACPP",
};

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Launch(launch) => match launch.run().await {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("Error: {err:?}");
                ExitCode::FAILURE
            }
        },
        CommandKind::Plan(cmd) => install_cmd(cli.instrumentation, cmd.execute()).await,
        CommandKind::Install(cmd) => install_cmd(cli.instrumentation, cmd.execute()).await,
        CommandKind::Uninstall(cmd) => install_cmd(cli.instrumentation, cmd.execute()).await,
    }
}

async fn install_cmd(
    instrumentation: Instrumentation,
    fut: impl std::future::Future<Output = anyhow::Result<ExitCode>>,
) -> ExitCode {
    if let Err(err) = instrumentation.setup() {
        eprintln!("Error: {err:?}");
        return ExitCode::FAILURE;
    }
    exit(fut.await)
}

fn exit(result: anyhow::Result<ExitCode>) -> ExitCode {
    match result {
        Ok(code) => code,
        Err(err) => {
            eprintln!("Error: {err:?}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Parser)]
#[clap(name = "llamacpp-launcher", version, about = "Launch Pi against a local llama.cpp router")]
struct Cli {
    #[clap(flatten)]
    instrumentation: Instrumentation,

    #[clap(subcommand)]
    command: CommandKind,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    /// Load a model on the router and start a coding agent
    Launch(Launch),
    /// Describe what this install would do
    Plan(Plan),
    /// Install the router, Pi, and this launcher
    Install(Install),
    /// Undo a previous install, following its receipt
    Uninstall(Uninstall),
}

#[derive(Debug, Parser)]
struct Plan {
    #[clap(long)]
    explain: bool,
    #[clap(long)]
    json: bool,
    #[clap(flatten)]
    planner: Router,
}

#[async_trait::async_trait]
impl CommandExecute for Plan {
    async fn execute(self) -> anyhow::Result<ExitCode> {
        cli::subcommand::plan::run(&APP, self.planner, self.explain, self.json).await
    }
}

#[derive(Debug, Parser)]
struct Install {
    #[clap(flatten)]
    planner: Router,
}

#[async_trait::async_trait]
impl CommandExecute for Install {
    async fn execute(self) -> anyhow::Result<ExitCode> {
        llamacpp::install::apply(&APP, self.planner).await
    }
}

#[derive(Debug, Parser)]
struct Uninstall {
    /// Receipt to follow (defaults to the planner's receipt path)
    #[clap(long)]
    receipt: Option<PathBuf>,
    #[clap(flatten)]
    planner: Router,
}

#[async_trait::async_trait]
impl CommandExecute for Uninstall {
    async fn execute(self) -> anyhow::Result<ExitCode> {
        let receipt = self.receipt.unwrap_or_else(|| self.planner.receipt_path());
        llamacpp::install::revert(&APP, receipt).await
    }
}

#[derive(Debug, Parser)]
struct Launch {
    /// Coding agent to start. Only `pi` is supported.
    agent: String,
    /// Model to load. Only `flint` is supported; Q4 / Q8 / BF16 is picked from RAM.
    #[clap(long)]
    model: String,
    #[clap(long, default_value = DEFAULT_HOST)]
    host: String,
    #[clap(long, default_value_t = DEFAULT_PORT)]
    port: u16,
    /// Context length passed to the router when loading
    #[clap(long)]
    ctx: Option<u32>,
    /// Offload every layer to the GPU. Detected from video memory when omitted:
    /// a card with at least 8 GiB does, a smaller one stays on the CPU.
    #[clap(long)]
    gpu: bool,
    /// Do not offload, even when the card has enough video memory
    #[clap(long, conflicts_with = "gpu")]
    cpu: bool,
    /// Do not write ~/.pi/agent/models.json or settings.json
    #[clap(long)]
    no_configure: bool,
    /// Leave a router we started running after Pi exits
    #[clap(long)]
    keep_server: bool,
    #[clap(flatten)]
    planner: Router,
    /// Arguments after `--` are passed to Pi
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    pi_args: Vec<String>,
}

impl Launch {
    fn offload(&self) -> bool {
        if self.cpu {
            return false;
        }
        if self.gpu {
            return true;
        }
        Machine::detect().offload()
    }

    async fn run(self) -> anyhow::Result<()> {
        if self.agent != "pi" {
            anyhow::bail!("only `pi` is supported (got `{}`)", self.agent);
        }
        if !self.model.eq_ignore_ascii_case("flint") {
            anyhow::bail!("only `flint` is supported (got `{}`)", self.model);
        }
        llamacpp::install::ensure(&APP, self.planner.clone()).await?;
        let offload = self.offload();
        let resolved = llamacpp::catalog::flint(offload)?;
        eprintln!("flint {}", resolved.file);
        write_preset(&resolved.preset(self.ctx.unwrap_or(2048), offload)).await?;
        let pi = which_pi()?;

        // Always start our own llama-server so it reads this launch's preset.
        // Reusing a listener would ignore hf-repo / n-gpu-layers we just wrote.
        let port = if tcp_open(&self.host, self.port).await {
            let free = free_port(&self.host).await?;
            eprintln!(
                "{host}:{port} is in use; using {host}:{free}",
                host = self.host,
                port = self.port,
            );
            free
        } else {
            self.port
        };
        let mut child = self.start_router(port).await?;
        let base = common::router_url(&self.host, port);
        let result = async {
            wait_loaded(&base, &mut child).await?;
            self.launch_pi(&base, &pi, &resolved).await
        }
        .await;
        if !self.keep_server {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        result
    }

    async fn start_router(&self, port: u16) -> anyhow::Result<Child> {
        let server = which_server()?;
        let models = common::models_dir();
        tokio::fs::create_dir_all(&models)
            .await
            .with_context(|| format!("creating {}", models.display()))?;
        let log = common::dir().join("llama-server.log");
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .with_context(|| format!("opening {}", log.display()))?;
        // Pi is an Ink TUI. Inheriting this process's stderr writes llama-server's
        // request log into the composer. Keep it in a file instead.
        let child = Command::new(&server)
            .arg("--host")
            .arg(&self.host)
            .arg("--port")
            .arg(port.to_string())
            .arg("--models-dir")
            .arg(&models)
            .arg("--models-preset")
            .arg(common::preset_path())
            .arg("--models-max")
            .arg("1")
            .arg("-ngl")
            .arg(if self.offload() { "999" } else { "0" })
            .stdin(Stdio::null())
            .stdout(log_file.try_clone()?)
            .stderr(log_file)
            .spawn()
            .with_context(|| format!("spawning {}", server.display()))?;
        eprintln!("llama-server log: {}", log.display());

        let base = common::router_url(&self.host, port);
        for _ in 0..50 {
            if probe(&base).await {
                return Ok(child);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        anyhow::bail!("router on {base} did not become ready");
    }

    async fn launch_pi(
        &self,
        base: &str,
        pi: &Path,
        resolved: &llamacpp::catalog::Resolved,
    ) -> anyhow::Result<()> {
        if !self.no_configure {
            configure_pi(base, &resolved.id, self.ctx.map(u64::from)).await?;
        }
        let mut args: Vec<String> = self
            .pi_args
            .iter()
            .filter(|arg| arg.as_str() != "--")
            .cloned()
            .collect();
        // `--model` is resolved against models.json before the llama.cpp
        // extension fetches the router catalog, so the id has to be declared
        // there. Pass the provider too: a bare id can match a logged-in
        // provider (xai) first and then fail.
        if !args.iter().any(|arg| arg == "--model" || arg == "-m") {
            args.insert(0, resolved.id.clone());
            args.insert(0, "--model".into());
        }
        if !args.iter().any(|arg| arg == "--provider") {
            args.insert(0, PI_PROVIDER.into());
            args.insert(0, "--provider".into());
        }
        eprintln!("exec {} {}", pi.display(), args.join(" "));
        let status = Command::new(pi)
            .args(&args)
            .status()
            .await
            .with_context(|| format!("running {}", pi.display()))?;
        if !status.success() {
            anyhow::bail!("pi exited with {status}");
        }
        Ok(())
    }
}

async fn write_preset(body: &str) -> anyhow::Result<()> {
    let path = common::preset_path();
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    tokio::fs::write(&path, body).await?;
    Ok(())
}

fn which_server() -> anyhow::Result<PathBuf> {
    let installed = common::llama_server_path();
    if installed.is_file() {
        return Ok(installed);
    }
    which("llama-server").context("llama-server is not installed")
}

fn which_pi() -> anyhow::Result<PathBuf> {
    let installed = common::pi_path();
    if common::pi_installed() {
        return Ok(installed);
    }
    which("pi").context("pi is not installed")
}

fn which(name: &str) -> anyhow::Result<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(anyhow!("`{name}` not on PATH"))
}

/// True only when the listener is llama-server. `/health` on that binary is JSON
/// (`{"status":"ok"}` once a model is up, or a models object in router mode).
async fn probe(base: &str) -> bool {
    let Ok(body) = http(base, "GET", "/health", None).await else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<Value>(body.trim()) else {
        return false;
    };
    let status = value.get("status").and_then(|v| v.as_str());
    status == Some("ok") || value.get("models").is_some() || status == Some("no slot available")
}

/// llama-server's `/health` is `{"status":"ok"}` only after a model is up.
/// A models object means the router is listening, not that the GGUF is ready.
async fn model_ready(base: &str) -> bool {
    let Ok(body) = http(base, "GET", "/health", None).await else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<Value>(body.trim()) else {
        return false;
    };
    matches!(
        value.get("status").and_then(|v| v.as_str()),
        Some("ok") | Some("no slot available")
    )
}

async fn wait_loaded(base: &str, child: &mut Child) -> anyhow::Result<()> {
    eprintln!("waiting for the model");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2 * 60 * 60);
    loop {
        if let Some(status) = child.try_wait()? {
            anyhow::bail!("llama-server exited with {status}");
        }
        if model_ready(base).await {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("model on {base} did not finish loading");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn tcp_open(host: &str, port: u16) -> bool {
    TcpStream::connect((host, port)).await.is_ok()
}

async fn free_port(host: &str) -> anyhow::Result<u16> {
    for port in (9931..9991).chain(18080..18100) {
        if !tcp_open(host, port).await {
            return Ok(port);
        }
    }
    anyhow::bail!("no free port for llama-server on {host}")
}

async fn configure_pi(base: &str, model: &str, context: Option<u64>) -> anyhow::Result<()> {
    // Pi resolves `--model` before its llama.cpp extension talks to the router,
    // so the model has to be in models.json or the flag exits with "not found".
    // An entry named `llama.cpp` is merged into the builtin as long as it does
    // not set `apiKey`.
    let dir = dirs_pi_agent();
    tokio::fs::create_dir_all(&dir).await?;

    let mut cfg = json!({
        "id": model,
        "name": model,
        "input": ["text"],
        "reasoning": false,
    });
    if let Some(context) = context.filter(|n| *n > 0) {
        cfg["contextWindow"] = json!(context);
    }

    let models_path = dir.join("models.json");
    let mut config = object(read_json(&models_path).await);
    let mut providers = match config.remove("providers") {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    };
    providers.remove("ollama");
    providers.remove("local");
    providers.insert(
        PI_PROVIDER.into(),
        json!({
            "baseUrl": format!("{base}/v1"),
            "api": "openai-completions",
            "authHeader": false,
            "models": [cfg],
        }),
    );
    config.insert("providers".into(), Value::Object(providers));
    write_json(&models_path, &Value::Object(config)).await?;

    let auth_path = dir.join("auth.json");
    let mut auth = object(read_json(&auth_path).await);
    auth.insert(
        PI_PROVIDER.into(),
        json!({
            "type": "api_key",
            "key": "local",
            "env": { "LLAMA_BASE_URL": base },
        }),
    );
    auth.remove("ollama");
    auth.remove("local");
    write_json(&auth_path, &Value::Object(auth)).await?;

    let settings = dir.join("settings.json");
    let mut doc = object(read_json(&settings).await);
    doc.insert("defaultProvider".into(), json!(PI_PROVIDER));
    doc.insert("defaultModel".into(), json!(model));
    write_json(&settings, &Value::Object(doc)).await?;
    Ok(())
}

fn dirs_pi_agent() -> PathBuf {
    common::home_dir().join(".pi").join("agent")
}

fn object(value: Option<Value>) -> Map<String, Value> {
    match value {
        Some(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

async fn read_json(path: &Path) -> Option<Value> {
    let text = tokio::fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&text).ok()
}

async fn write_json(path: &Path, value: &Value) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    let mut tmp = path.to_path_buf();
    tmp.set_extension("json.tmp");
    let mut file = tokio::fs::File::create(&tmp).await?;
    file.write_all(text.as_bytes()).await?;
    file.write_all(b"\n").await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}

/// A small HTTP/1.0 client. The installer already depends on tokio, and the
/// launcher should not pull reqwest in again.
async fn http(
    base: &str,
    method: &str,
    path: &str,
    body: Option<&Value>,
) -> anyhow::Result<String> {
    let (host, port) = parse_base(base)?;
    let payload = body.map(ToString::to_string).unwrap_or_default();
    let mut request = format!(
        "{method} {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\nAccept: application/json\r\n"
    );
    if body.is_some() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", payload.len()));
    }
    request.push_str("\r\n");
    request.push_str(&payload);

    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect((host.as_str(), port)))
        .await
        .with_context(|| format!("connecting to {base}"))?
        .with_context(|| format!("connecting to {base}"))?;
    stream.write_all(request.as_bytes()).await?;
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut buf))
        .await
        .context("reading the router")?
        .context("reading the router")?;
    let text = String::from_utf8_lossy(&buf);
    let (head, resp_body) = match text.split_once("\r\n\r\n") {
        Some(parts) => parts,
        None => ("", text.as_ref()),
    };
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if !(200..300).contains(&status) {
        anyhow::bail!("{method} {path} returned {status}: {resp_body}");
    }
    Ok(resp_body.to_string())
}

fn parse_base(base: &str) -> anyhow::Result<(String, u16)> {
    let rest = base
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("only http:// router URLs are supported ({base})"))?;
    let hostport = rest.split('/').next().unwrap_or(rest);
    match hostport.split_once(':') {
        Some((host, port)) => Ok((host.to_string(), port.parse().context("router port")?)),
        None => Ok((hostport.to_string(), 80)),
    }
}
