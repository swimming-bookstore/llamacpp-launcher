//! Install a llama.cpp router, Pi, and `llamacpp-launcher`.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Context;
use installer::action::base::{
    CreateDirectory, FetchAndUnpackTarball, InstallBinary, InstallTree, ReleaseArchive,
    RequirePaths,
};
use installer::action::{Action, StatefulAction};
use installer::planner::{diff_from_default, Planner};
use url::Url;

use crate::common;

/// Install the router, Pi, and this launcher.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, clap::Parser)]
pub struct Router {
    /// This launcher's directory: receipt, unpack, preset, log. Defaults to
    /// `~/.llamacpp-launcher`. Binaries still go to `~/.local/bin`.
    #[clap(long, env = "LLAMACPP_DIR")]
    pub dir: Option<PathBuf>,

    /// Staged llama.cpp release tarball (skips the download)
    #[clap(long, env = "LLAMACPP_LLAMA_TARBALL")]
    pub llama_tarball: Option<PathBuf>,

    /// SHA-256 of the llama.cpp release. Required; a release with no checksum is refused.
    /// Left empty, the checksum of the build selected for this machine is used.
    #[clap(long, env = "LLAMACPP_LLAMA_SHA256")]
    pub llama_sha256: Option<String>,

    /// llama.cpp backend: `cpu`, `rocm` (AMD), or `vulkan`. Detected when omitted.
    #[clap(long, env = "LLAMACPP_BACKEND")]
    pub backend: Option<String>,

    /// Staged Pi release tarball (skips the download)
    #[clap(long, env = "LLAMACPP_PI_TARBALL")]
    pub pi_tarball: Option<PathBuf>,

    /// SHA-256 of the Pi release. Empty uses the pin in catalog.json.
    #[clap(long, env = "LLAMACPP_PI_SHA256")]
    pub pi_sha256: Option<String>,

    /// Where this launcher binary is, so it can be copied into `~/.local/bin`
    #[clap(long, env = "LLAMACPP_LAUNCHER_BIN")]
    #[serde(alias = "installer_bin")]
    pub launcher_bin: Option<PathBuf>,
}

#[async_trait::async_trait]
#[typetag::serde(name = "router")]
impl Planner for Router {
    async fn default() -> anyhow::Result<Self> {
        Ok(Self {
            dir: None,
            llama_tarball: None,
            llama_sha256: None,
            backend: None,
            pi_tarball: None,
            pi_sha256: None,
            launcher_bin: None,
        })
    }

    async fn plan(&self) -> anyhow::Result<Vec<StatefulAction<Box<dyn Action>>>> {
        let dir = self.dir.clone().unwrap_or_else(common::dir);
        let bin = common::bin_dir();
        let scratch = dir.join("unpack");
        let llama_unpack = scratch.join("llama");
        let llama = common::llama_dir();
        let pi_unpack = scratch.join("pi");
        let server = common::llama_server_path();
        // Beside package.json. `pi --version` reads that file from the directory
        // of the executable and prints 0.0.0 when it is missing.
        let pi = bin.join("pi-coding-agent").join("pi");
        let launcher = bin.join("llamacpp-launcher");
        let models = dir.join("models");

        let asset = self.llama_asset()?;
        let llama_sha = self.llama_sha256.clone().unwrap_or_else(|| asset.sha256.clone());
        let llama_archive = match &self.llama_tarball {
            Some(path) => ReleaseArchive::from_path(path, &llama_sha)?,
            None => {
                eprintln!("llama.cpp {} ({})", asset.name, self.selected_backend()?.label());
                ReleaseArchive::from_url(Url::parse(&asset.url)?, &llama_sha)?
            }
        };
        let pins = crate::catalog::pins()?;
        let pi_sha = self
            .pi_sha256
            .clone()
            .unwrap_or_else(|| pins.pi.sha256.clone());
        let pi_archive = match &self.pi_tarball {
            Some(path) => ReleaseArchive::from_path(path, &pi_sha)?,
            None => ReleaseArchive::from_url(Url::parse(&pins.pi.url)?, &pi_sha)?,
        };

        let launcher_src = self.launcher_bin.clone().unwrap_or_else(|| {
            std::env::current_exe().unwrap_or_else(|_| PathBuf::from("llamacpp-launcher"))
        });

        let mut actions: Vec<StatefulAction<Box<dyn Action>>> = vec![
            CreateDirectory::plan(&bin, None, None, Some(0o755), false).await?.boxed(),
            CreateDirectory::plan(&dir, None, None, Some(0o755), false).await?.boxed(),
            CreateDirectory::plan(&models, None, None, Some(0o755), false).await?.boxed(),
            FetchAndUnpackTarball::plan(llama_archive, &llama_unpack).await?.boxed(),
            FetchAndUnpackTarball::plan(pi_archive, &pi_unpack).await?.boxed(),
            // llama-server is dynamically linked against the .so files that ship
            // next to it. InstallTree matches a directory suffix, so the tree is
            // the parent of `llama-server` rather than a file name.
            InstallContainingTree::plan(&llama_unpack, "llama-server", &llama)
                .await?
                .boxed(),
            // Pi's compiled binary reads package.json from the directory of the
            // executable. InstallBinary alone leaves that file behind, and the
            // binary then reports 0.0.0. The release tree goes under
            // `~/.local/bin/pi-coding-agent/`, and `install` puts the real `pi`
            // binary in that same directory. Matching `pi/pi` skips the example
            // packages, which also contain package.json.
            InstallTree::plan(
                &pi_unpack,
                "pi/pi",
                bin.join("pi-coding-agent"),
                None::<String>,
                None::<String>,
            )
            .await?
            .boxed(),
            InstallBinary::plan(&pi_unpack, "pi", &pi, None::<String>).await?.boxed(),
            // The launcher is this binary. Copy it into the scratch tree so
            // InstallBinary can find it by name. A symlink would be skipped.
            StageFile::plan(&launcher_src, scratch.join("self").join("llamacpp-launcher"))
                .await?
                .boxed(),
            InstallBinary::plan(scratch.join("self"), "llamacpp-launcher", &launcher, None::<String>)
                .await?
                .boxed(),
        ];

        // The router is not a service. `llamacpp-launcher` starts it in the
        // background, so install never writes a unit file and never needs root.

        actions.push(
            RequirePaths::plan(
                "the router, Pi, and the launcher",
                vec![
                    (server.clone(), "llama-server".into()),
                    (pi.clone(), "pi".into()),
                    (launcher.clone(), "llamacpp-launcher".into()),
                ],
                "re-run `llamacpp-launcher launch pi`; the release archives must contain those binaries",
            )
            .await?
            .boxed(),
        );

        Ok(actions)
    }

    fn settings(&self) -> anyhow::Result<HashMap<String, serde_json::Value>> {
        let mut map = HashMap::new();
        map.insert(
            "dir".into(),
            json_string(self.dir.clone().unwrap_or_else(common::dir).display()),
        );
        map.insert(
            "llama_tarball".into(),
            match &self.llama_tarball {
                Some(path) => json_string(path.display()),
                None => serde_json::Value::Null,
            },
        );
        map.insert(
            "pi_tarball".into(),
            match &self.pi_tarball {
                Some(path) => json_string(path.display()),
                None => serde_json::Value::Null,
            },
        );
        Ok(map)
    }

    async fn configured_settings(&self) -> anyhow::Result<HashMap<String, serde_json::Value>> {
        diff_from_default(self).await
    }

    fn receipt_path(&self) -> PathBuf {
        self.dir
            .clone()
            .unwrap_or_else(common::dir)
            .join("install-receipt.json")
    }

    async fn platform_check(&self) -> anyhow::Result<()> {
        if !cfg!(target_os = "linux") {
            anyhow::bail!("this installer targets Linux");
        }
        Ok(())
    }
}

impl Router {
    fn llama_asset(&self) -> anyhow::Result<crate::gpu::Asset> {
        self.selected_backend()?.asset()
    }

    fn selected_backend(&self) -> anyhow::Result<crate::gpu::Backend> {
        match self.backend.as_deref() {
            Some(name) => parse_backend(name),
            None => Ok(crate::gpu::Machine::detect().backend),
        }
    }
}

fn parse_backend(name: &str) -> anyhow::Result<crate::gpu::Backend> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(crate::gpu::Backend::Cpu),
        "rocm" | "amd" | "hip" => Ok(crate::gpu::Backend::Rocm),
        "vulkan" => Ok(crate::gpu::Backend::Vulkan),
        other => anyhow::bail!("unknown llama.cpp backend `{other}` (cpu, rocm, vulkan)"),
    }
}

fn json_string(value: impl ToString) -> serde_json::Value {
    serde_json::Value::String(value.to_string())
}

/// Copy the directory that contains `file` (llama-server and its `.so` files).
#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
struct InstallContainingTree {
    search_root: PathBuf,
    file: String,
    dest: PathBuf,
}

impl InstallContainingTree {
    async fn plan(
        search_root: impl Into<PathBuf>,
        file: impl Into<String>,
        dest: impl Into<PathBuf>,
    ) -> anyhow::Result<StatefulAction<Self>> {
        Ok(installer::action::StatefulAction::uncompleted(Self {
            search_root: search_root.into(),
            file: file.into(),
            dest: dest.into(),
        }))
    }
}

#[async_trait::async_trait]
#[typetag::serde(name = "install_containing_tree")]
impl Action for InstallContainingTree {
    fn tracing_synopsis(&self) -> String {
        format!(
            "Install the directory containing `{}` to `{}`",
            self.file,
            self.dest.display()
        )
    }

    fn tracing_span(&self) -> tracing::Span {
        tracing::span!(tracing::Level::DEBUG, "install_containing_tree")
    }

    fn execute_description(&self) -> Vec<installer::action::ActionDescription> {
        vec![installer::action::ActionDescription::new(
            self.tracing_synopsis(),
            vec![],
        )]
    }

    async fn execute(&mut self) -> anyhow::Result<()> {
        let source = installer::util::find_file(&self.search_root, &self.file)
            .and_then(|path| path.parent().map(PathBuf::from))
            .with_context(|| {
                format!(
                    "the release archive did not contain `{}`",
                    self.file
                )
            })?;
        installer::util::remove_dir_all_if_exists(&self.dest)
            .await
            .with_context(|| format!("removing `{}`", self.dest.display()))?;
        tokio::fs::create_dir_all(&self.dest)
            .await
            .with_context(|| format!("creating `{}`", self.dest.display()))?;
        installer::util::copy_dir_all(&source, &self.dest).await?;
        Ok(())
    }

    fn revert_description(&self) -> Vec<installer::action::ActionDescription> {
        vec![installer::action::ActionDescription::new(
            format!("Remove `{}`", self.dest.display()),
            vec![],
        )]
    }

    async fn revert(&mut self) -> anyhow::Result<()> {
        installer::util::remove_dir_all_if_exists(&self.dest)
            .await
            .with_context(|| format!("removing `{}`", self.dest.display()))?;
        Ok(())
    }
}

/// Copy one file into the scratch tree so a later [`InstallBinary`] can find it.
///
/// Revert removes only the staged copy. The installed binary is removed by
/// [`InstallBinary`]'s own revert.
#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
struct StageFile {
    source: PathBuf,
    dest: PathBuf,
}

impl StageFile {
    async fn plan(
        source: impl Into<PathBuf>,
        dest: impl Into<PathBuf>,
    ) -> anyhow::Result<StatefulAction<Self>> {
        Ok(installer::action::StatefulAction::uncompleted(Self {
            source: source.into(),
            dest: dest.into(),
        }))
    }
}

#[async_trait::async_trait]
#[typetag::serde(name = "stage_file")]
impl Action for StageFile {
    fn tracing_synopsis(&self) -> String {
        format!(
            "Stage `{}` at `{}`",
            self.source.display(),
            self.dest.display()
        )
    }

    fn tracing_span(&self) -> tracing::Span {
        tracing::span!(tracing::Level::DEBUG, "stage_file")
    }

    fn execute_description(&self) -> Vec<installer::action::ActionDescription> {
        vec![installer::action::ActionDescription::new(
            self.tracing_synopsis(),
            vec![],
        )]
    }

    async fn execute(&mut self) -> anyhow::Result<()> {
        if let Some(parent) = self.dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::copy(&self.source, &self.dest)
            .await
            .map_err(|err| anyhow::anyhow!("staging `{}`: {err}", self.source.display()))?;
        Ok(())
    }

    fn revert_description(&self) -> Vec<installer::action::ActionDescription> {
        vec![installer::action::ActionDescription::new(
            format!("Remove staged `{}`", self.dest.display()),
            vec![],
        )]
    }

    async fn revert(&mut self) -> anyhow::Result<()> {
        if self.dest.is_file() {
            tokio::fs::remove_file(&self.dest).await.ok();
        }
        Ok(())
    }
}
