//! Which llama.cpp build to fetch, and whether the router should use the GPU.
//!
//! An AMD card gets the ROCm build from the pinned release. Vulkan is the
//! fallback when there is no ROCm device but a Vulkan ICD is present. Anything
//! else stays on the CPU build. A card with a lot of video memory offloads
//! layers; a small one does not, so a 22 GB model is not forced into a few GB
//! of VRAM.

use std::path::Path;

/// Offload when the largest card has at least this much video memory.
const LARGE_VRAM: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asset {
    pub name: String,
    pub sha256: String,
    pub url: String,
}

/// Backend the router is built and started for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    /// AMD ROCm. Needs `/dev/kfd` or a ROCm install, not just an AMD PCI id.
    Rocm,
    Vulkan,
}

impl Backend {
    pub fn asset(self) -> anyhow::Result<Asset> {
        crate::catalog::pins()?.llama(self.label())
    }

    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Rocm => "rocm",
            Backend::Vulkan => "vulkan",
        }
    }
}

/// What the machine can run, and how the start command should use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Machine {
    pub backend: Backend,
    /// Largest video memory seen, in bytes. `None` when it could not be read.
    pub vram: Option<u64>,
}

impl Machine {
    pub fn detect() -> Self {
        let gpus = drm_gpus();
        let amd = gpus.iter().any(|gpu| gpu.amd);
        let vram = gpus.iter().filter_map(|gpu| gpu.vram).max();
        let backend = if !cfg!(target_arch = "x86_64") {
            Backend::Cpu
        } else if amd && rocm_present() {
            Backend::Rocm
        } else if vulkan_usable() && gpus.iter().any(|gpu| !gpu.virtio) {
            Backend::Vulkan
        } else {
            Backend::Cpu
        };
        Self { backend, vram }
    }

    /// `-ngl 999` only when a card has enough video memory to hold a real model.
    /// A few GB of VRAM, or an unknown amount, stays on the CPU path.
    pub fn offload(self) -> bool {
        self.backend != Backend::Cpu && self.vram.is_some_and(|bytes| bytes >= LARGE_VRAM)
    }
}

#[derive(Debug)]
struct DrmGpu {
    amd: bool,
    virtio: bool,
    vram: Option<u64>,
}

fn drm_gpus() -> Vec<DrmGpu> {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };
    let mut gpus = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // `card0`, not `card0-HDMI-A-1`.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let vendor = read_hex(&device.join("vendor"));
        let vram = std::fs::read_to_string(device.join("mem_info_vram_total"))
            .ok()
            .and_then(|text| text.trim().parse().ok());
        gpus.push(DrmGpu {
            amd: vendor == Some(0x1002),
            virtio: vendor == Some(0x1af4),
            vram,
        });
    }
    gpus
}

fn read_hex(path: &Path) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok()
}

/// ROCm user space is present. The PCI id alone is not enough: the ROCm binary
/// will not start on a machine with no `/dev/kfd`.
fn rocm_present() -> bool {
    Path::new("/dev/kfd").exists()
        || Path::new("/opt/rocm").is_dir()
        || std::env::var_os("ROCM_PATH").is_some_and(|value| Path::new(&value).is_dir())
}

fn vulkan_usable() -> bool {
    ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"]
        .iter()
        .any(|dir| {
            std::fs::read_dir(dir)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .any(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_asset_is_the_pinned_amd_build() {
        let asset = Backend::Rocm.asset().unwrap();
        assert!(asset.name.contains("rocm"));
        assert!(asset.url.contains("/b11852/"));
        assert_eq!(asset.sha256.len(), 64);
    }

    #[test]
    fn small_or_unknown_vram_stays_on_cpu_layers() {
        let small = Machine {
            backend: Backend::Vulkan,
            vram: Some(2 * 1024 * 1024 * 1024),
        };
        let unknown = Machine {
            backend: Backend::Rocm,
            vram: None,
        };
        let large = Machine {
            backend: Backend::Rocm,
            vram: Some(16 * 1024 * 1024 * 1024),
        };
        assert!(!small.offload());
        assert!(!unknown.offload());
        assert!(large.offload());
        assert!(!Machine { backend: Backend::Cpu, vram: Some(LARGE_VRAM) }.offload());
    }
}
