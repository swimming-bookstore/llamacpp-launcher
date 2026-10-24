//! Pins and `--model` resolution.
//!
//! `catalog.json` is bundled. `LLAMACPP_CATALOG` replaces it without a rebuild.
//! A path is loaded as-is. A Hugging Face `user/repo[:quant]` or a short name
//! becomes a preset; llama.cpp downloads the GGUF itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;

const BUNDLED: &str = include_str!("../catalog.json");

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Pins {
    pub llama: LlamaRelease,
    pub pi: PiRelease,
    pub models: HashMap<String, ModelPin>,
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
pub struct ModelPin {
    pub id: String,
    pub repo: String,
    pub file: String,
}

impl Pins {
    pub fn model(&self, name: &str) -> Option<&ModelPin> {
        self.models
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, pin)| pin)
    }

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
    Ok(pins)
}

fn require_sha256(field: &str, value: &str) -> anyhow::Result<()> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(())
    } else {
        anyhow::bail!("catalog.json {field} must be 64 hex chars")
    }
}

#[derive(Debug, Clone)]
pub struct Resolved {
    pub id: String,
    pub source: Source,
}

#[derive(Debug, Clone)]
pub enum Source {
    HuggingFace { repo: String, file: Option<String> },
    Path(PathBuf),
}

impl Resolved {
    /// Router preset. llama.cpp fetches Hugging Face pins; a path is loaded as-is.
    pub fn preset(&self, ctx: u32, offload: bool) -> String {
        let mut text = crate::common::preset_globals(ctx, offload);
        text.push_str(&format!("\n[{}]\nload-on-startup = true\n", self.id));
        match &self.source {
            Source::HuggingFace { repo, file } => {
                text.push_str(&format!("hf-repo = {repo}\n"));
                if let Some(file) = file {
                    text.push_str(&format!("hf-file = {file}\n"));
                }
            }
            Source::Path(path) => {
                text.push_str(&format!("model = {}\n", path.display()));
            }
        }
        text
    }
}

/// Turn `--model` into a router id and a source llama.cpp can load.
pub fn resolve(model: &str, models_dir: &Path) -> anyhow::Result<Resolved> {
    if looks_like_path(model) {
        let path = PathBuf::from(model);
        if !path.is_file() {
            anyhow::bail!("model file not found: {}", path.display());
        }
        return Ok(Resolved {
            id: stem(&path, model),
            source: Source::Path(path),
        });
    }

    if let Some((repo, quant)) = hf_ref(model) {
        let repo = match quant {
            Some(quant) => format!("{repo}:{quant}"),
            None => repo,
        };
        return Ok(Resolved {
            id: hf_id(&repo),
            source: Source::HuggingFace { repo, file: None },
        });
    }

    if let Some(pin) = pins()?.model(model) {
        return Ok(Resolved {
            id: pin.id.clone(),
            source: Source::HuggingFace {
                repo: pin.repo.clone(),
                file: Some(pin.file.clone()),
            },
        });
    }

    if let Some(existing) = find_local(models_dir, model)? {
        return Ok(Resolved {
            id: stem(&existing, model),
            source: Source::Path(existing),
        });
    }

    anyhow::bail!("unknown model `{model}`")
}

fn stem(path: &Path, fallback: &str) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(fallback)
        .to_string()
}

fn hf_id(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    name.replace(':', "-")
}

fn looks_like_path(model: &str) -> bool {
    model.ends_with(".gguf")
        || model.contains('/') && !model.contains(':') && Path::new(model).exists()
        || model.starts_with('.')
        || model.starts_with('~')
}

/// `user/repo` or `user/repo:Q4_K_M`. A local path is not this.
fn hf_ref(model: &str) -> Option<(String, Option<String>)> {
    if model.starts_with('.') || model.starts_with('/') || model.starts_with('~') {
        return None;
    }
    let (repo, quant) = match model.split_once(':') {
        Some((repo, quant)) => (repo, Some(quant.to_string())),
        None => (model, None),
    };
    let mut parts = repo.split('/');
    let user = parts.next()?;
    let name = parts.next()?;
    if parts.next().is_some() || user.is_empty() || name.is_empty() {
        return None;
    }
    if !repo
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
    {
        return None;
    }
    Some((repo.to_string(), quant))
}

fn find_local(models_dir: &Path, query: &str) -> anyhow::Result<Option<PathBuf>> {
    let needle = query.to_ascii_lowercase();
    let mut hits = Vec::new();
    let dir = match std::fs::read_dir(models_dir) {
        Ok(dir) => dir,
        Err(_) => return Ok(None),
    };
    for entry in dir {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.ends_with(".gguf") && name.contains(&needle) {
            hits.push(path);
        }
    }
    match hits.len() {
        0 => Ok(None),
        1 => Ok(hits.pop()),
        _ => anyhow::bail!("more than one model matches `{query}`"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_catalog_has_the_pinned_releases() {
        let pins = parse(BUNDLED).unwrap();
        assert_eq!(pins.llama.tag, "b11852");
        let rocm = pins.llama("rocm").unwrap();
        assert!(rocm.name.contains("rocm"));
        assert!(rocm.url.contains("/b11852/"));
        assert_eq!(rocm.sha256.len(), 64);
        assert!(pins.pi.url.contains("pi-mono"));
        let flint = pins.model("flint").expect("flint pin");
        assert_eq!(flint.repo, "ggml-org/Flint-GGUF");
        assert!(flint.file.ends_with(".gguf"));
    }

    #[test]
    fn flint_preset_asks_llama_cpp_to_fetch() {
        let resolved = resolve("flint", Path::new("/no/models")).unwrap();
        let preset = resolved.preset(2048, false);
        assert!(preset.contains("[Flint-Q4_K_M]"));
        assert!(preset.contains("hf-repo = ggml-org/Flint-GGUF"));
        assert!(preset.contains("hf-file = Flint-Q4_K_M.gguf"));
        assert!(!preset.contains("http"));
    }
}
