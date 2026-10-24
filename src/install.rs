//! Apply and revert a [`Router`] plan without the framework's sudo escalation.
//!
//! The framework's `install` and `uninstall` commands call `ensure_root` before
//! they do anything else. This product installs under the user's home, the way
//! `ollama` keeps its own files, so those entry points are not usable. The plan,
//! the actions, and the receipt are still the framework's.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use installer::cli::App;
use installer::InstallPlan;

use crate::planner::Router;

/// Plan and apply. The steps are not printed; a launch should look like
/// `ollama launch`, not like an installer.
pub async fn apply(app: &App, planner: Router) -> anyhow::Result<ExitCode> {
    let mut plan = installer::cli::plan_with(app, planner).await?;
    execute(&mut plan, app.version).await?;
    Ok(ExitCode::SUCCESS)
}

/// Ensure the router, Pi, and launcher are installed. A launch calls this when
/// the binaries are missing; it is the same plan `llamacpp-launcher install` runs.
pub async fn ensure(app: &App, planner: Router) -> anyhow::Result<()> {
    if crate::common::llama_server_path().is_file()
        && crate::common::pi_installed()
        && crate::common::launcher_path().is_file()
    {
        return Ok(());
    }
    let mut plan = installer::cli::plan_with(app, planner).await?;
    execute(&mut plan, app.version).await?;
    Ok(())
}

async fn execute(plan: &mut InstallPlan, version: &str) -> anyhow::Result<()> {
    plan.install(None, version).await
}

/// Undo a receipt without escalating.
pub async fn revert(app: &App, receipt: PathBuf) -> anyhow::Result<ExitCode> {
    let contents = tokio::fs::read_to_string(&receipt).await.with_context(|| {
        format!(
            "reading the receipt `{}`; nothing was installed from here",
            receipt.display()
        )
    })?;
    let mut plan: InstallPlan = serde_json::from_str(&contents).with_context(|| {
        format!(
            "parsing the receipt `{}`; it may have been written by an incompatible version",
            receipt.display()
        )
    })?;
    plan.uninstall(None, app.version).await?;
    Ok(ExitCode::SUCCESS)
}
