//! Pins. `catalog.json` is bundled. `LLAMACPP_CATALOG` replaces it without a rebuild.
//! Flint is `ggml-org/Flint-GGUF`; the quant (Q4, Q8, BF16) is picked from RAM
//! (plus VRAM when layers go to the GPU) and passed as `-hf repo:quant`.

use std::path::Path;

use anyhow::Context;

use crate::gpu::Machine;

const BUNDLED: &str = include_str!("../catalog.json");
/// Headroom for the OS, KV cache, and runtime. The GGUF itself is memory-mapped.
const HEADROOM_GIB: u64 = 8;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Pins {
    pub llama: LlamaRelease,
    pub pi: PiRelease,
    pub flint: Flint,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaRelease {
    pub tag: String,
    pub url: String,
    pub assets: LlamaAssets,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaAssets {
    pub cpu: LlamaFile,
    pub rocm: LlamaFile,
    pub vulkan: LlamaFile,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaFile {
    pub name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PiRelease {
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Flint {
    pub repo: String,
    pub quants: Vec<FlintQuant>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct FlintQuant {
    pub hf: String,
    pub gib: u64,
}

impl Pins {
    pub fn llama(&self, backend: &str) -> anyhow::Result<crate::gpu::Asset> {
        let file = match backend {
            "cpu" => &self.llama.assets.cpu,
            "rocm" => &self.llama.assets.rocm,
            "vulkan" => &self.llama.assets.vulkan,
            other => anyhow::bail!("no llama.cpp asset for `{other}`"),
        };
        let url = self
            .llama
            .url
            .replace("{tag}", &self.llama.tag)
            .replace("{name}", &file.name);
        Ok(crate::gpu::Asset {
            name: file.name.clone(),
            sha256: file.sha256.clone(),
            url,
        })
    }
}

/// Bundled pins, or the file at `LLAMACPP_CATALOG`.
pub fn pins() -> anyhow::Result<Pins> {
    match std::env::var_os("LLAMACPP_CATALOG") {
        Some(path) => load_file(Path::new(&path)),
        None => parse(BUNDLED).context("parsing bundled catalog.json"),
    }
}

fn load_file(path: &Path) -> anyhow::Result<Pins> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading catalog {}", path.display()))?;
    parse(&text).with_context(|| format!("parsing catalog {}", path.display()))
}

fn parse(text: &str) -> anyhow::Result<Pins> {
    let pins: Pins = serde_json::from_str(text).context("catalog.json")?;
    require_sha256("pi.sha256", &pins.pi.sha256)?;
    if pins.pi.url.is_empty() {
        anyhow::bail!("catalog.json pi.url is missing");
    }
    for (backend, file) in [
        ("cpu", &pins.llama.assets.cpu),
        ("rocm", &pins.llama.assets.rocm),
        ("vulkan", &pins.llama.assets.vulkan),
    ] {
        if file.name.is_empty() {
            anyhow::bail!("catalog.json llama.assets.{backend}.name is missing");
        }
        require_sha256(&format!("llama.assets.{backend}.sha256"), &file.sha256)?;
    }
    if pins.flint.repo.is_empty() || pins.flint.quants.is_empty() {
        anyhow::bail!("catalog.json flint is missing");
    }
    Ok(pins)
}

fn require_sha256(field: &str, value: &str) -> anyhow::Result<()> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(())
    } else {
        anyhow::bail!("catalog.json {field} must be 64 hex chars")
    }
}

/// `-hf` name llama.cpp should fetch for this machine.
pub fn flint_hf(offload: bool) -> anyhow::Result<String> {
    flint_hf_for(Machine::detect(), offload)
}

fn flint_hf_for(machine: Machine, offload: bool) -> anyhow::Result<String> {
    let pins = pins()?;
    let quant = pick(&pins.flint.quants, machine, offload)
        .context("catalog.json flint.quants is empty")?;
    Ok(format!("{}:{}", pins.flint.repo, quant.hf))
}

fn pick(quants: &[FlintQuant], machine: Machine, offload: bool) -> Option<&FlintQuant> {
    let ram = machine.ram.unwrap_or(0);
    let vram = if offload { machine.vram.unwrap_or(0) } else { 0 };
    let gib = (ram + vram) / (1024 * 1024 * 1024);
    quants
        .iter()
        .find(|quant| gib >= quant.gib + HEADROOM_GIB)
        .or_else(|| quants.last())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Backend;

    #[test]
    fn bundled_catalog_has_the_pinned_releases() {
        let pins = parse(BUNDLED).unwrap();
        assert_eq!(pins.llama.tag, "b11852");
        let rocm = pins.llama("rocm").unwrap();
        assert!(rocm.name.contains("rocm"));
        assert!(rocm.url.contains("/b11852/"));
        assert_eq!(rocm.sha256.len(), 64);
        assert!(pins.pi.url.contains("pi-mono"));
        assert_eq!(pins.flint.repo, "ggml-org/Flint-GGUF");
    }

    #[test]
    fn more_memory_picks_a_larger_flint() {
        let q4 = Machine {
            backend: Backend::Cpu,
            vram: None,
            ram: Some(32 * 1024 * 1024 * 1024),
        };
        let q8 = Machine {
            backend: Backend::Rocm,
            vram: Some(24 * 1024 * 1024 * 1024),
            ram: Some(32 * 1024 * 1024 * 1024),
        };
        let bf16 = Machine {
            backend: Backend::Rocm,
            vram: Some(48 * 1024 * 1024 * 1024),
            ram: Some(64 * 1024 * 1024 * 1024),
        };
        assert_eq!(flint_hf_for(q4, false).unwrap(), "ggml-org/Flint-GGUF:Q4_K_M");
        assert_eq!(flint_hf_for(q8, true).unwrap(), "ggml-org/Flint-GGUF:Q8_0");
        assert_eq!(flint_hf_for(bf16, true).unwrap(), "ggml-org/Flint-GGUF:BF16");
    }
}
