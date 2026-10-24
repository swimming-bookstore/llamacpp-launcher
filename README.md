# llamacpp-launcher

```
llamacpp-launcher launch pi --model flint
```

Installs llama.cpp under `~/.llamacpp-launcher/llama` and Pi into `~/.local/bin` if needed, runs `llama-server -hf ggml-org/Flint-GGUF:Q4_K_M` (or Q8_0 / BF16 from RAM), then Pi. No root.

State lives in `~/.llamacpp-launcher`. Pins are in `catalog.json`.

`llamacpp-launcher uninstall` removes what it installed.
