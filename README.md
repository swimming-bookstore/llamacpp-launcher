# llamacpp-launcher

```
llamacpp-launcher launch pi --model flint
```

Installs llama.cpp under `~/.llamacpp-launcher/llama` and Pi into `~/.local/bin` if needed, starts the router, and runs Pi. No root.

The model is always Flint. Q4, Q8, or BF16 is picked from RAM (and VRAM when the GPU is used).

State lives in `~/.llamacpp-launcher`. Pins are in `catalog.json`.

`llamacpp-launcher uninstall` removes what it installed.
