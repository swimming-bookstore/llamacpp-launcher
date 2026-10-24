//! Paths shared by the installer and `llamacpp-launcher`.
//!
//! `llamacpp-launcher pi` writes the router model under Pi's builtin
//! `llama.cpp` provider and then execs `pi`.
//!
//! Pi 1.4.0 resolves `--model` before the llama.cpp extension fills its
//! catalog, so a model that exists only on the router is "not found". The
//! model is declared in models.json, which is loaded first and merged into
//! the builtin provider.

use std::path::PathBuf;

pub const DEFAULT_HOST: &str = "127.0.0.1";
/// llama-server's own default. 8080 is commonly taken (dev servers), and the
/// router must not be confused with whatever else is listening there.
pub const DEFAULT_PORT: u16 = 9931;
/// Provider Pi uses for the router. This is the builtin `llama.cpp` id. A
/// `models.json` entry with the same id is merged into that builtin when it
/// does not set `apiKey` (an empty key fails schema check and hides the
/// builtin). Not `ollama`: that name is what the footer showed.
pub const PI_PROVIDER: &str = "llama.cpp";

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// This launcher's files: receipt, unpack scratch, preset, log, local GGUFs.
/// Not llama.cpp's cache. `LLAMACPP_DIR` overrides it.
pub fn dir() -> PathBuf {
    std::env::var_os("LLAMACPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".llamacpp-launcher"))
}

/// `~/.local/bin`, which is already on a normal user's PATH. The framework's
/// `install` puts the binaries here as real files.
pub fn bin_dir() -> PathBuf {
    std::env::var_os("LLAMACPP_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".local").join("bin"))
}

/// Unpacked llama.cpp tree. `llama-server` is dynamically linked against the
/// `.so` files that ship next to it, so the whole release stays together.
pub fn llama_dir() -> PathBuf {
    dir().join("llama")
}

pub fn llama_server_path() -> PathBuf {
    llama_dir().join("llama-server")
}

pub fn pi_path() -> PathBuf {
    bin_dir().join("pi-coding-agent").join("pi")
}

/// True when the installed `pi` can read its version. The binary looks for
/// `package.json` in the directory of the executable; without that file it
/// reports `0.0.0`.
pub fn pi_installed() -> bool {
    let bin = pi_path();
    if !bin.is_file() {
        return false;
    }
    let resolved = std::fs::canonicalize(&bin).unwrap_or(bin);
    resolved
        .parent()
        .is_some_and(|dir| dir.join("package.json").is_file())
}

pub fn launcher_path() -> PathBuf {
    bin_dir().join("llamacpp-launcher")
}

pub fn models_dir() -> PathBuf {
    dir().join("models")
}

/// Preset applied to every model the router loads. Weight repacking copies the
/// whole GGUF into RAM; a model larger than the machine (Flint is 22 GB) then
/// fails to allocate. Memory-mapping keeps the weights on disk.
pub fn preset_path() -> PathBuf {
    dir().join("models.ini")
}

pub fn preset_globals(ctx: u32, offload: bool) -> String {
    // `999` asks llama-server to put every layer it can on the GPU. `0` keeps
    // the weights on the CPU, which is what a card with little video memory needs.
    let layers = if offload { 999 } else { 0 };
    format!(
        "\
[*]
no-repack = on
ctx-size = {ctx}
n-gpu-layers = {layers}
"
    )
}

pub fn receipt_path() -> PathBuf {
    dir().join("install-receipt.json")
}

pub fn router_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}")
}
