# Nightage on Termux (EXPERIMENTAL)

Runs in the Termux terminal on Android; no app is needed. Not an official platform and **not yet tried on a real phone** - see
`docs/PORTABILITY.md` for what is verified (only through fakes) and what to test. Install Termux from F-Droid (the Play Store build is outdated).

## Install

```sh
pkg update && pkg install -y git
git clone <this repository> nightage && cd nightage
sh install-termux.sh           # asks about a remote model server; --yes skips questions; --dry-run shows the plan
nightage --profile phone doctor
nightage --profile phone
```

The script installs `python`, `git`, `ripgrep`, a virtualenv (`~/.nightage-venv`) with only pure-Python dependencies (typer, rich,
pyyaml, httpx, prompt_toolkit) and a launcher `$PREFIX/bin/nightage`. `sqlite-vec`, `ruff` and `coverage` are skipped on purpose
(`pyproject.toml` marks them `sys_platform != 'android'`): memory recall uses keywords, the code agent skips lint and the coverage
gate (`pkg install ruff` brings the lint back). The project itself is not pip-installed because its build backend needs a compiled
wheel.

## Where does the model run?

- **On the home PC (recommended on most phones).** Put the PC's Tailscale address in `ollama.host` (the script can do it), run
  `nightage --profile phone doctor` to see the latency. Details: `docs/PORTABILITY.md`, "Remote gateway".
- **On the phone.** `pkg install ollama`, `ollama serve &`, `ollama pull qwen3.5:2b` (UNVERIFIED: a package exists in termux-packages, speed and
  stability on your device are unknown). The `phone` profile picks `qwen3.5:0.8b / 2b / 4b` by RAM and a 4k-16k context.

## Tips

- `termux-setup-storage` gives access to Downloads/Documents under `~/storage` (the doctor shows a hint).
- `termux-wake-lock` stops Android from suspending Termux during a long run.
- Close the chat normally (not by swiping Termux away) so `ollama.unload_on_exit` can free the model.
- Narrow screens: the UI has a narrow layout below 60 columns; if glyphs look broken run with `NIGHTAGE_PLAIN=1`.
