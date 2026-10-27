# Portability: which systems work, what is experimental

Nightage was built on NixOS with a desktop GPU. This page says what is known to work elsewhere. Anything marked **UNVERIFIED** was
written defensively and tested only with fakes (fake `/proc`, fake environment, fake `shutil.which`); nobody has run it on that system yet.

`nightage doctor` shows the platform row (OS, distro, init system, RAM) and suggests `--profile phone` on Termux or under 6 GB of RAM.
The same facts are available in code as `agent.sysinfo.detect()` (never raises).

## Feature table

`ok` = used and tested on it (or tested through fakes and no known gap), `fb` = works with a fallback, `-` = not available, `?` = UNVERIFIED.

| Feature | NixOS / systemd distro | Other Linux (no systemd) | Termux (Android) | macOS | Windows |
|---|---|---|---|---|---|
| Terminal chat, doctor, settings | ok | ok | ? (light deps tested on Linux only) | ? (CI runs it) | ? (CI runs it; guided `install.ps1`, see INSTALL.md) |
| Local Ollama | ok | ok | ? `pkg install ollama` (see below) | ? | ? |
| Remote Ollama (`ollama.host` / `OLLAMA_HOST`) | ok | ok | ok in principle (HTTP only) - UNVERIFIED on a phone | ok | ok |
| Vector memory (`sqlite-vec`) | ok | ok | fb: keyword recall (no wheel expected) | ? | ? |
| Coding agent | ok | ok | fb: no ruff / coverage gate unless installed | ? | ? |
| Status panel (`/proc`: CPU, RAM, load) | ok | ok | fb: blanks where `/proc/stat` is blocked | - | - |
| Background worker (`systemctl --user`) | ok | - message "systemctl unavailable" | - | - | - |
| Autopilot keeps PC awake (`systemd-inhibit`) | ok | fb: not protected against suspend | fb: same | - | - |
| Dual-GPU Vulkan routing (`ollama-vk`) | ok | needs your own start command (`AGENT_VULKAN_START_CMD`) | - (skipped when the server is remote) | - | - |
| Web GUI / web apps | ok (`web` dependency group) | ok | - not offered by the phone profile | ? | ? |
| Host page: GPU (nvidia-smi), suspend | ok | fb: GPU row empty, suspend answers 501 | fb | - | - |
| "Game running" pause (`/proc/*/cmdline`) | ok | ok | fb: never pauses | - | - |
| WhatsApp plugin | ok | ok | - disabled by the phone profile | ? | ? |

## Capability guards (audited)

| Place | Assumption | Now |
|---|---|---|
| `agent/panel.py` | `/proc/stat`, `/proc/meminfo`, `/proc/loadavg` readable | each returns a neutral value when unreadable (Android blocks `/proc/stat` for apps) |
| `agent/web/routes/host.py` | `systemctl suspend`, `nvidia-smi`, `/sys/class/thermal` | suspend answers HTTP 501 with a reason when there is no systemd; the others already returned `None` |
| `agent/web/routes/ops.py`, `ops_control.py`, `agent/main.py worker` | `systemctl --user`, `journalctl` | already guarded (empty answer / 503 "not available") - now covered by tests |
| `agent/web/routes/whatsapp_persona.py` | `pkill` | no-op when `pkill` is missing |
| `agent/llm.py` | `systemctl --user` for the `ollama-vk` unit, `~/.local/bin/ollama-vk` | guarded by `shutil.which`; the whole Vulkan route is skipped when the server is remote |
| `agent/codeagent/supervisor.py` | `systemd-inhibit` | already guarded by `which`; run is just not protected against suspend |
| `agent/governor.py` | `/proc/*/cmdline`, `nvidia-smi` | already guarded (returns `None`) |
| `agent/coding/verify` | `ruff`, `coverage` installed | lint / coverage gate are skipped with a note when absent |
| `agent/memory/recall.py` | `sqlite-vec` loadable | keyword recall over active facts instead of nothing |
| `agent/repoaccess.py` | harmless read prefixes | **safety file, additive only:** `/data/data/com.termux/files/usr/` added next to `/usr/`; nothing removed |
| `agent/tools/files.py` drive listing | `/media`, `/run/media`, `/mnt` | answers "No mounted drives found" (already graceful); Termux shared storage is `~/storage` |

Known gaps (not fixed): macOS and Windows were not audited beyond not crashing on import; `agent/tools/system.py` uses `ps` and
`/proc/meminfo`; `install.sh` supports Debian/Ubuntu/Arch/Fedora/openSUSE only (Termux has `install-termux.sh`); Nix-specific
paths in the safety classifier (`/nix/store/`) are harmless elsewhere.

## Profiles

`profile: default | phone` in `config.yaml`, env `NIGHTAGE_PROFILE`, or `nightage --profile phone <command>` (the option sets the env for
that run). The profile is an **overlay at load time**: `built-in defaults < profile overlay < your config.yaml`. It is never written
back, and any key you set yourself wins. `profile` in the loaded config always shows the effective profile.

`phone` (EXPERIMENTAL) sets, for a machine with 4 GB RAM and a local Ollama (values by RAM in brackets):

| Key | Value | Why |
|---|---|---|
| `models.chat` / `coding` / `reasoning` / `unrestricted` | `qwen3.5:0.8b` (<3 GB), `qwen3.5:2b` (<6 GB or unknown), `qwen3.5:4b` | names exist on ollama.com/library/qwen3.5/tags (checked 2026-10-26); per-tag tool support is not stated there - UNVERIFIED |
| `models.background` | `qwen3.5:0.8b` | |
| `models.embed` | `all-minilm` (46 MB) | bge-m3 is ~1.2 GB; a fresh phone DB has no embedding-size history (embedding dimension is coupled to the DB: do not sync a phone DB with a PC DB) |
| `models.max_ctx_ceiling`, `chat_num_ctx` | 4096 (<3 GB), 6144 (<4), 8192 (<6), 12288 (<8), 16384 | `agent.ctxbudget` caps every model by the ceiling |
| `ollama.unload_on_exit` | `true` | models are evicted when you close the tool |
| `ollama.keep_alive` | `5m` (default `30m`) | frees RAM sooner between messages |
| `memory.recall_k` / `context_char_budget` | 3 / min(24000, 2 x ctx) | |
| `tools.max_iterations` | 20 (60) | |
| `tools.disabled` | `deep_research`, `generate_image`, `send_album`, `send_file`, `get_media_info`, `make_diagram`, `wa_*` | heavy / media / WhatsApp; `agent.tools.registry.configure` |
| `ui.mood_emotes` | `false` | no kaomoji; plain ASCII glyphs are switched on (`NIGHTAGE_PLAIN=1`) when the terminal is not UTF-8; the launcher hides the Web GUI entries |
| `autonomy.max_lookups_per_turn` / `fail_streak` / `repeat_same_error` / `identical_calls` / `plan` | 1 / 2 / 2 / 2 / `false` (3 / 3 / 2 / 3 / `true` lookups 3) | chat stuck watcher (`agent/autonomy.py`); the three thresholds are new config keys with the old values as defaults |
| `coding.loop_note` / `loop_repeat` / `loop_abort` | 1 / 2 / 3 (2 / 3 / 4) | near-duplicate call ladder |
| `coding.family_note` / `family_abort` | 3 / 5 (5 / 8) | |
| `coding.run_spam_note` / `run_spam_abort` | 6 / 9 (10 / 14) | |
| `coding.same_tool_note` / `same_tool_abort` | 5 / 7 (8 / 12) | |
| `coding.alt_note` / `alt_abort` | 2 / 3 (3 / 4) | |
| `coding.read_repeat_note` / `read_repeat_abort` | 3 / 5 (4 / 8) | |
| `coding.chant_hits` / `chant_think_hits` | 8 / 6 (14 / 10) | repetition inside one reply |
| `coding.malformed_abort` / `auto_extend_max` / `announce_max` / `stray_looks` / `look_block_at` | 3 / 3 / 4 / 5 / 10 (5 / 15 / 8 / 8 / 16) | |
| `coding.stage_max_steps` / `stage_max_minutes` | 80 / 10 (180 / 25) | |
| `coding.num_predict` / `think_predict` / `sprawl_tokens` / `sprawl_quiet` | 2048 / 1500 / 3000 / 1000 (8192 / 4000 / 6000 / 2000) | shorter runaway generations |
| `background.max_ram_gb` | 2 (16) | |

With a **remote gateway** (`ollama.host` set, or `OLLAMA_HOST`) the profile does not touch the models (the home PC runs its normal
ones) and uses a 16384 context ceiling; the phone's RAM is ignored for the context budget because the model does not run there.

The loop catcher itself (`agent/codeagent/loopguard.py`, `engine.py`) and the chat turn engine are unchanged: the profile only moves
thresholds through the existing `coding.*` keys (read by `Engine._limit`) and the new `autonomy.*` keys.

## Remote gateway: UI on the phone, model on the PC

1. On the PC: make Ollama listen on the network (`OLLAMA_HOST=0.0.0.0 ollama serve`; with systemd set it in the service environment) and install Tailscale; note the PC's Tailscale address (`tailscale ip -4`) or MagicDNS name.
2. On the phone: Tailscale app connected to the same tailnet.
3. In `config.yaml`: `ollama: {host: "100.x.y.z:11434"}`, or `export OLLAMA_HOST=100.x.y.z:11434` (the environment wins).
4. `nightage --profile phone doctor` shows "remote running", the median round trip of `/api/version` (warns above 1.5 s) and, if it is down, what to check.

Precedence: `OLLAMA_HOST` > `ollama.host` > `http://localhost:11434`. A remote host serves every model; the local dual-GPU Vulkan server is
not used then. `unload_on_exit` still unloads the model on the PC when you leave. Ollama has no authentication: only expose it on a private
network such as a tailnet, never to the open internet.

## Native Ollama on the phone (UNVERIFIED)

The termux-packages repository has an `ollama` package (`packages/ollama/build.sh`, version 0.40.11 when checked; excluded for 32-bit
arm and i686, so aarch64/x86_64 only; Vulkan backend). So `pkg install ollama` then `ollama serve &` is the expected route and several third-party
guides describe it. Not verified by us on hardware. Expect: CPU or Vulkan inference, a 1-4 B model at a few tokens per second, a model download of 1-3 GB, and
Android killing background processes under memory pressure. Nightage now manages this server itself (`agent/ollama_local.py`,
`nightage ollama start|stop|status|list|pull|rm`, `nightage pull <model>`; autostart and stop-on-exit are the config keys `ollama.autostart` /
`ollama.stop_on_exit`); the walkthrough is [TERMUX.md](TERMUX.md). Rooting is optional and not needed by anything here (it would only help
with stopping the OOM killer and with `swapon`).

## What to try on a real phone

1. `sh install-termux.sh --dry-run`, then for real. Does `python -m venv` and the pip install of the five light dependencies work?
2. `nightage --profile phone doctor`: platform row, RAM figure, storage row.
3. Remote gateway over Tailscale: latency row and a chat.
4. `nightage ollama start`, `nightage pull qwen3.5:2b` (percent bar), then a chat: does the server start from Termux and survive a screen-off, and is `qwen3.5:2b` usable with tools? Does `nightage ollama status` show the server after Android killed it (phantom process killer)?
5. Narrow screen (about 40-50 columns): `/help`, the home menu, a tool call, a long answer.
6. Close the tool (/quit, Ctrl+C, closing the Termux session): `ollama ps` should be empty and the server stopped (`unload_on_exit`, `stop_on_exit`).

## Windows audit (UNVERIFIED on real Windows; what is proven here and what is not)

Proven on Linux by `tests/test_windows_import.py`: a child interpreter blocks `fcntl termios pty pwd grp resource curses tty`, flips
`sys.platform` to `win32`, imports EVERY module of the package and runs `--help`, `version`, `doctor --list`, `demo --help`. The helpers
live in `agent/osport.py`. Not proven: anything that actually runs on a Windows kernel (the GitHub Actions matrix `cross-platform` is the first real signal).

| File | Issue on Windows | Fix |
|---|---|---|
| `agent/codeagent/tuismoke.py` | imports `fcntl`, `termios`, `select`; needs a pty | guarded import; `run_smoke` answers with a clear "needs a POSIX pty" error |
| `agent/coding/verify/runner.py` | `import resource` (memory cap) | optional import; no memory cap on Windows, the timeout still applies |
| `agent/codeagent/session.py`, `sessionstore.py` | `fcntl.flock` folder lock | `osport.try_lock` / `unlock` (`msvcrt.locking` on Windows) |
| `agent/codeagent/tools.py`, `supervisor.py` | `os.killpg(SIGKILL)` | `osport.kill_group` (`taskkill /T /F` on Windows) |
| `agent/codeagent/tools.py`, `agent/tools/shell.py`, `agent/coding/tools.py`, `coding_cli.py` | run commands with `bash -lc` | `osport.shell_argv`: uses `bash` if present (Git Bash / WSL), else a clear message recommending WSL; chat, memory and demo do not need it |
| `agent/codeagent/policy.py` | `os.getuid()`, `/tmp` rules | no uid on Windows: not "ours"; policy file under `%APPDATA%` |
| `agent/update.py`, `gpumode.py`, `repoaccess.py`, `ui/dock.py`, `codeagent/sessionstore.py`, `eventlog.py`, `undo.py`, `config.py`, `plugins.py`, `web/auth.py` | XDG paths (`~/.config`, `~/.local/state`, `~/.local/share`) | `osport.config_home/state_home/data_home`: Linux/macOS unchanged, Windows `%APPDATA%` / `%LOCALAPPDATA%` |
| `agent/textio.py` | `select.select` on stdin | skipped on Windows (a paste is one line) |
| `agent/sysinfo.py`, `ctxbudget.py` | RAM only from `/proc/meminfo` (0 on Windows, wrong model tier) | `osport.native_ram_mb` (`GlobalMemoryStatusEx` / `sysctl hw.memsize`) |
| console encoding | cp1252 crashes on glyphs/emoji, mis-reads UTF-8 files | `agent/__init__.py` -> `osport.windows_utf8_setup`: stdout/stderr UTF-8 and one restart with `-X utf8` when started as `nightage` (`NIGHTAGE_NO_REEXEC=1` skips); `ui/style.plain_mode` falls back to ASCII on a non-UTF-8 stream |
| `agent/web/routes/host.py` | `os.getloadavg` missing | `AttributeError` handled |
| `panel.py`, `governor.py`, `tools/system.py` | `/proc`, `ps` | were already graceful (neutral values); unchanged |
| web routes (`whatsapp_*`, `ops*`) | `pkill`, `/proc`, `systemctl`, `start_new_session` | server-side Linux features, not offered on Windows; unchanged |
| not fixed | `shlex` quoting in `agent/safety.py` (backslash paths); the code agent's own tools assume POSIX paths; symlink-based features | use WSL for the coding agent; documented |
