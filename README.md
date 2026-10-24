# llamacpp-launcher

```
llamacpp-launcher launch pi --model flint
```

Installs llama.cpp under `~/.llamacpp-launcher/llama` and Pi into `~/.local/bin` if needed, starts `llama-server` in the background if it is not already up, then Pi. Same command again only starts Pi. No root.

State lives in `~/.llamacpp-launcher`. Pins are in `catalog.json`.

`llamacpp-launcher start --model flint` starts llama-server in the background without Pi. Same flags as launch. Already up → no second server.

`llamacpp-launcher stop` stops the llama-server this launcher started.

`llamacpp-launcher uninstall` removes what it installed.
