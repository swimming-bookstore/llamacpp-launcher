# llamacpp-launcher

```
llamacpp-launcher launch pi --model flint
```

Installs llama.cpp and Pi into `~/.local/bin` if needed, starts the router, and runs Pi. No root.

State lives in `~/.llamacpp-launcher`. Pins (llama.cpp, Pi, short model names) are in `catalog.json`.

`llamacpp-launcher uninstall` removes what it installed.
