//! Paths shared by the installer and `llamacpp-launcher`.
//!
//! `llamacpp-launcher launch pi` writes the model under Pi's builtin
//! `llama.cpp` provider and then execs `pi`.
//!
//! Pi 1.4.0 resolves `--model` before the llama.cpp extension fills its
//! catalog, so a model that exists only on the router is "not found". The
//! model is declared in models.json, which is loaded first and merged into
//! the builtin provider.

use std::path::PathBuf;

pub const DEFAULT_HOST: &str = "127.0.0.1";
/// llama-server's own default. 8080 is commonly taken (dev servers).
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

/// This launcher's files: receipt, unpack scratch, log.
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

pub fn receipt_path() -> PathBuf {
    dir().join("install-receipt.json")
}

pub fn pid_path() -> PathBuf {
    dir().join("llama-server.pid")
}

pub fn log_path() -> PathBuf {
    dir().join("llama-server.log")
}

pub fn router_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}")
}
