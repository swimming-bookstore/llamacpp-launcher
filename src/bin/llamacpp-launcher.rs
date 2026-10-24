//! `llamacpp-launcher launch pi --model flint`
//!
//! Installs llama.cpp under `~/.llamacpp-launcher/llama` and Pi into
//! `~/.local/bin` if they are missing, starts the router, and execs Pi.
//! `plan`, `install`, and `uninstall` are the same binary.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};
use std::time::Duration;

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use installer::cli::{self, arg::Instrumentation, CommandExecute};
use installer::planner::Planner;
use serde_json::{json, Map, Value};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
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
        CommandKind::Start(start) => match start.run().await {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("Error: {err:?}");
                ExitCode::FAILURE
            }
        },
        CommandKind::Stop => match stop_server() {
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
    /// Start llama-server in the background without Pi
    Start(Start),
    /// Stop the llama-server this launcher started
    Stop,
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
struct Server {
    /// Model to load. Only `flint`; Q4 / Q8 / BF16 is picked from RAM.
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
    #[clap(flatten)]
    planner: Router,
}

impl Server {
    fn offload(&self) -> bool {
        if self.cpu {
            return false;
        }
        if self.gpu {
            return true;
        }
        Machine::detect().offload()
    }

    async fn ensure(&self) -> anyhow::Result<String> {
        if !self.model.eq_ignore_ascii_case("flint") {
            anyhow::bail!("only `flint` is supported (got `{}`)", self.model);
        }
        llamacpp::install::ensure(&APP, self.planner.clone()).await?;
        let hf = llamacpp::catalog::flint_hf(self.offload())?;
        let base = common::router_url(&self.host, self.port);
        self.ensure_server(&base, &hf).await?;
        Ok(base)
    }

    /// One llama-server on host:port. Already up → reuse. Already starting →
    /// wait. Otherwise start it in the background and leave it running.
    async fn ensure_server(&self, base: &str, hf: &str) -> anyhow::Result<()> {
        if health_ok(base).await {
            return Ok(());
        }
        if llama_server_running() {
            eprintln!("waiting for llama-server on {base}");
            return wait_ready(base, None, 0, None).await;
        }
        if tcp_open(&self.host, self.port).await {
            anyhow::bail!("{base} is in use by something that is not llama-server");
        }
        eprintln!("llama-server -hf {hf}");
        let log = common::log_path();
        if let Some(dir) = log.parent() {
            tokio::fs::create_dir_all(dir).await?;
        }
        let offset = tokio::fs::metadata(&log).await.map(|m| m.len()).unwrap_or(0);
        let child = self.start_server(hf, &log).await?;
        let pid = child.id().context("llama-server pid")?;
        write_pid(pid)?;
        wait_ready(base, Some(&log), offset, Some(pid)).await
    }

    async fn start_server(&self, hf: &str, log: &Path) -> anyhow::Result<Child> {
        let server = which_server()?;
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .with_context(|| format!("opening {}", log.display()))?;
        let mut cmd = Command::new(&server);
        cmd.arg("--host")
            .arg(&self.host)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("-hf")
            .arg(hf)
            .arg("-ngl")
            .arg(if self.offload() { "999" } else { "0" });
        if let Some(ctx) = self.ctx {
            cmd.arg("--ctx-size").arg(ctx.to_string());
        }
        cmd.stdin(Stdio::null())
            .stdout(log_file.try_clone()?)
            .stderr(log_file)
            .process_group(0)
            .kill_on_drop(false)
            .spawn()
            .with_context(|| format!("spawning {}", server.display()))
    }
}

#[derive(Debug, Parser)]
struct Start {
    #[clap(flatten)]
    server: Server,
}

impl Start {
    async fn run(self) -> anyhow::Result<()> {
        let base = self.server.ensure().await?;
        eprintln!("llama-server on {base}");
        Ok(())
    }
}

#[derive(Debug, Parser)]
struct Launch {
    /// Coding agent to start. Only `pi` is supported.
    agent: String,
    #[clap(flatten)]
    server: Server,
    /// Do not write ~/.pi/agent/models.json or settings.json
    #[clap(long)]
    no_configure: bool,
    /// Arguments after `--` are passed to Pi
    #[clap(trailing_var_arg = true, allow_hyphen_values = true)]
    pi_args: Vec<String>,
}

impl Launch {
    async fn run(self) -> anyhow::Result<()> {
        if self.agent != "pi" {
            anyhow::bail!("only `pi` is supported (got `{}`)", self.agent);
        }
        let base = self.server.ensure().await?;
        let pi = which_pi()?;
        let model = match loaded_model(&base).await {
            Ok(id) => id,
            Err(_) => llamacpp::catalog::flint_hf(self.server.offload())?,
        };
        self.launch_pi(&base, &pi, &model).await
    }

    async fn launch_pi(&self, base: &str, pi: &Path, model: &str) -> anyhow::Result<()> {
        if !self.no_configure {
            configure_pi(base, model, self.server.ctx.map(u64::from)).await?;
        }
        let mut args: Vec<String> = self
            .pi_args
            .iter()
            .filter(|arg| arg.as_str() != "--")
            .cloned()
            .collect();
        // `--model` is resolved against models.json before the llama.cpp
        // extension fetches the catalog, so the id has to be declared there.
        // Pass the provider too: a bare id can match a logged-in provider first.
        if !args.iter().any(|arg| arg == "--model" || arg == "-m") {
            args.insert(0, model.to_string());
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

fn write_pid(pid: u32) -> anyhow::Result<()> {
    let path = common::pid_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, format!("{pid}\n")).with_context(|| format!("writing {}", path.display()))
}

fn read_pid(path: &Path) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    text.trim().parse().ok()
}

fn our_llama_server(pid: u32) -> bool {
    let Ok(server) = which_server() else {
        return false;
    };
    let Ok(want) = std::fs::canonicalize(&server) else {
        return false;
    };
    process_exe(pid).is_some_and(|exe| exe == want)
}

fn stop_server() -> anyhow::Result<()> {
    let path = common::pid_path();
    let recorded = read_pid(&path);
    let mut stopped = false;
    if let Some(pid) = recorded {
        if our_llama_server(pid) {
            kill_group(pid);
            stopped = true;
        }
    }
    for pid in our_llama_server_pids() {
        kill_pid(pid);
        stopped = true;
    }
    let _ = std::fs::remove_file(&path);
    if stopped {
        eprintln!("stopped llama-server");
    } else {
        eprintln!("llama-server is not running");
    }
    Ok(())
}

fn our_llama_server_pids() -> Vec<u32> {
    let Ok(server) = which_server() else {
        return Vec::new();
    };
    let Ok(want) = std::fs::canonicalize(&server) else {
        return Vec::new();
    };
    let me = std::process::id();
    all_pids()
        .into_iter()
        .filter(|&pid| pid != me && process_exe(pid).is_some_and(|exe| exe == want))
        .collect()
}

fn kill_group(pid: u32) {
    unsafe {
        libc::killpg(pid as i32, libc::SIGTERM);
    }
    for _ in 0..20 {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    unsafe {
        libc::killpg(pid as i32, libc::SIGKILL);
    }
}

fn kill_pid(pid: u32) {
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    for _ in 0..20 {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    unsafe {
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

fn llama_server_running() -> bool {
    !our_llama_server_pids().is_empty()
}

fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

fn process_exe(pid: u32) -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_link(format!("/proc/{pid}/exe")).ok()
    }
    #[cfg(target_os = "macos")]
    {
        let mut buf = [0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
        let n = unsafe {
            libc::proc_pidpath(pid as i32, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as u32)
        };
        if n <= 0 {
            return None;
        }
        let path = std::str::from_utf8(&buf[..n as usize]).ok()?;
        Some(PathBuf::from(path))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = pid;
        None
    }
}

fn all_pids() -> Vec<u32> {
    #[cfg(target_os = "linux")]
    {
        let Ok(proc) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        proc.flatten()
            .filter_map(|entry| entry.file_name().to_str()?.parse().ok())
            .collect()
    }
    #[cfg(target_os = "macos")]
    {
        let needed = unsafe { libc::proc_listpids(libc::PROC_ALL_PIDS, 0, std::ptr::null_mut(), 0) };
        if needed <= 0 {
            return Vec::new();
        }
        let cap = (needed as usize / std::mem::size_of::<i32>()).saturating_add(32);
        let mut buf = vec![0i32; cap];
        let bytes = unsafe {
            libc::proc_listpids(
                libc::PROC_ALL_PIDS,
                0,
                buf.as_mut_ptr() as *mut libc::c_void,
                (buf.len() * std::mem::size_of::<i32>()) as i32,
            )
        };
        if bytes <= 0 {
            return Vec::new();
        }
        let n = bytes as usize / std::mem::size_of::<i32>();
        buf.into_iter()
            .take(n)
            .filter_map(|pid| (pid > 0).then_some(pid as u32))
            .collect()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Vec::new()
    }
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

/// `-hf` llama-server only answers `/health` with `{"status":"ok"}` after the
/// GGUF is loaded. A download keeps the port closed until then. New log bytes
/// are copied to our stderr so progress is visible; once ready, Pi owns the TTY.
async fn wait_ready(
    base: &str,
    log: Option<&Path>,
    mut offset: u64,
    pid: Option<u32>,
) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2 * 60 * 60);
    loop {
        if let Some(pid) = pid {
            if !pid_alive(pid) {
                anyhow::bail!("llama-server exited");
            }
        } else if !llama_server_running() && !health_ok(base).await {
            anyhow::bail!("llama-server on {base} exited");
        }
        if let Some(path) = log {
            offset = tail(path, offset).await;
        }
        if health_ok(base).await {
            if log.is_some() {
                let _ = io::stderr().write_all(b"\n");
            }
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("llama-server on {base} did not become ready");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn tail(path: &Path, offset: u64) -> u64 {
    let Ok(mut file) = tokio::fs::File::open(path).await else {
        return offset;
    };
    if file.seek(std::io::SeekFrom::Start(offset)).await.is_err() {
        return offset;
    }
    let mut buf = [0u8; 4096];
    let mut pos = offset;
    loop {
        match file.read(&mut buf).await {
            Ok(0) | Err(_) => return pos,
            Ok(n) => {
                let _ = io::stderr().write_all(&buf[..n]);
                let _ = io::stderr().flush();
                pos += n as u64;
            }
        }
    }
}

async fn health_ok(base: &str) -> bool {
    let Ok(body) = http(base, "GET", "/health", None).await else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<Value>(body.trim()) else {
        return false;
    };
    value.get("status").and_then(|v| v.as_str()) == Some("ok")
}

async fn loaded_model(base: &str) -> anyhow::Result<String> {
    let body = http(base, "GET", "/v1/models", None).await?;
    let value: Value = serde_json::from_str(body.trim()).context("parsing /v1/models")?;
    value
        .get("data")
        .and_then(|v| v.as_array())
        .and_then(|data| data.first())
        .and_then(|item| item.get("id"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .context("llama-server listed no models")
}

async fn tcp_open(host: &str, port: u16) -> bool {
    TcpStream::connect((host, port)).await.is_ok()
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
