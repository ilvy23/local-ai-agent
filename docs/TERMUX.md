# Nightage on your phone

Nightage runs in **Termux**, the terminal app for Android. No other app is needed. You can run the AI model **on the phone itself**
(path 1) or keep the model on your **PC at home** and use the phone as the remote screen (path 2).

> **Honest status: not yet tried on a real phone.** Everything here was written from documentation and tested with fakes (fake `pkg`,
> fake `ollama`, fake terminal widths). Lines marked **UNVERIFIED** are expectations, not observations. If something differs on your
> phone, `docs/PORTABILITY.md` lists what to report.

Install Termux from **F-Droid** (the Play Store build is outdated). Termux:API (optional, also F-Droid) adds `termux-wake-lock`.

Contents: [Path 1: everything on the phone](#path-1-everything-on-the-phone) - [Path 2: model on your PC](#path-2-model-on-your-pc-over-tailscale) -
[Commands](#commands-for-the-models-on-the-phone) - [Keeping it alive](#keeping-it-alive) - [Freeing the memory](#freeing-the-memory) -
[Speed](#how-fast-will-it-be) - [Troubleshooting](#troubleshooting) - [Uninstall](#uninstall)

## Path 1: everything on the phone

You need roughly: **6 GB of RAM or more** (4 GB works with the smallest models), **5 GB free space**, a Wi-Fi connection for the downloads.

### 1. Run the installer

Open Termux and paste the one line from [INSTALL.md](INSTALL.md) (it downloads the project and starts the phone installer), or, if you
already have the folder:

```sh
sh install-termux.sh
```

The installer is quiet and calm. It adapts to your screen width (a phone is about 40-60 columns), shows one bar for the **whole**
installation, and puts the noisy output of `pkg` and `pip` into a log file:

```
[████████░░░░░░░░░░░░] 38%  Step 4 of 8
Command and settings
  ✓ Command ready: nightage
  ? Use the phone profile? Small models,
    short context, frees memory when you
    leave. [Y/n]
```

The eight steps: check the phone, install `python`/`git`/`ripgrep`, make the Python environment, create the `nightage` command and
settings, install and start **Ollama**, download the **chat model**, download the **memory model**, run the health check.
Every question has a default (press Enter). It is safe to run again: it only does what is missing, and a download that was stopped resumes.

Useful options (`sh install-termux.sh --help` lists all):

| Option | Meaning |
|---|---|
| `--yes` | take the default answer to every question |
| `--dry-run` | show the plan, change nothing |
| `--verbose` | show the output of `pkg` / `pip` instead of logging it |
| `--no-ollama` | do not install Ollama on the phone (use path 2) |
| `--model qwen3.5:2b` | a different chat model than the recommended one |
| `--embed all-minilm` / `--no-embed` | choose the memory model, or none (keyword recall only) |
| `--profile default` | do not switch to the phone profile |

If a step fails, you see the **last 8 lines** of what it printed and the path of the log
(`~/.local/state/nightage/install.log`). Send that file when you report a problem.

### 2. What gets chosen for you

The installer picks **one** chat model from the phone's memory (the same tiers as `agent/profiles.py`):

| Phone memory | Chat model | Download (about, UNVERIFIED) |
|---|---|---|
| under 3 GB | `qwen3.5:0.8b` | 1.0 GB |
| 3 to 6 GB | `qwen3.5:2b` | 2.7 GB |
| 6 GB or more | `qwen3.5:4b` | 3.4 GB |

With 10 GB or more, `qwen3.5:9b` (about 6.6 GB) is possible too: it needs most of the memory and is slow, so it is not the default
(`nightage pull qwen3.5:9b`, then `nightage settings` to select it). Under 6 GB the installer **does not** put models on the phone by
default (it asks, default "no"): path 2 is usually better there.

The **memory model** helps Nightage recall things by meaning. You are asked whether you mostly chat in English: yes gives `all-minilm`
(46 MB), no gives `embeddinggemma:300m-qat-q4_0` (239 MB, 100+ languages). You can skip it; recall then works on keywords.

### 3. Start

```sh
nightage
```

The start screen shows **Ollama on this phone: running** or **not running** with the exact fix. Opening a chat starts Ollama by itself
if it is not running (`ollama.autostart`), and leaving the chat unloads the model and stops that Ollama again (`ollama.stop_on_exit`),
so the phone's RAM comes back. The first message after a start is slow while the model loads.

### 4. More models

```sh
nightage pull qwen3.5:2b          # percent, speed, ETA; Ctrl+C is safe, run the same command to continue
nightage ollama list              # what is installed
nightage ollama rm qwen3.5:2b     # free the space again
```

`nightage pull` refuses a model that obviously will not fit (memory or free space) and says why; `--force` overrides that.

## Path 2: model on your PC over Tailscale

The phone is the screen, the PC does the thinking. Best for phones under 6 GB and for bigger models.

1. Install [Tailscale](https://tailscale.com) on the PC and on the phone, sign in to the same account.
2. On the PC, let Ollama listen on the network: set `OLLAMA_HOST=0.0.0.0` for the Ollama service and restart it. Ollama has **no
   password**: only do this on a private network like your tailnet, never on the open internet.
3. Note the PC's Tailscale address (for example `100.64.0.5`).
4. On the phone: `NIGHTAGE_REMOTE_HOST=100.64.0.5 sh install-termux.sh`, or answer "yes" to "Use Ollama on another machine?" in the
   installer, or write `ollama: {host: 100.64.0.5}` into `config.yaml`.
5. `nightage doctor` shows the latency to the PC. Nightage never starts or stops anything on the PC.

With a remote host, the models stay the PC's business: the phone profile only trims the client side.

## Commands for the models on the phone

| Command | What it does |
|---|---|
| `nightage ollama status` | running or not, address, installed models, the fix when it is not |
| `nightage ollama start` | start Ollama in the background (own process group, waits until it answers) |
| `nightage ollama stop` | stop the Ollama **that Nightage started** (never one you or the system started) |
| `nightage ollama list` / `rm <model>` | installed models / delete one |
| `nightage pull <model>` | download with a percent bar (same as `nightage ollama pull`) |
| `nightage doctor --only system,ollama,models` | health check of the phone, Ollama and models |

**Models need no file editing.** Every role (chat, notes, code, reasoning, memory) is `auto` on a fresh install: Nightage uses the models that
are really installed and says so in one line. If you have one chat model and one memory model, everything uses them. `nightage models`
shows who uses what, `nightage models use <model>` switches every chat role, `nightage models embed <model>` the memory model. If you
did choose a model in `config.yaml` that is not installed, Nightage asks once whether to use an installed one instead (and only saves it after
you say yes). `nightage doctor` shows which `config.yaml` is loaded (on Termux: `~/.local/share/nightage/src/config.yaml`).

Settings (`nightage settings`, or `config.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `ollama.autostart` | `true` | start Ollama when chat/code/research/web begins and nothing answers on this machine (needs the `ollama` program; never when `ollama.host` is another machine) |
| `ollama.stop_on_exit` | `true` | when you leave, stop the Ollama Nightage started (only that one) |
| `ollama.unload_on_exit` | `true` | free the loaded models when you leave |
| `ollama.keep_alive` | `5m` (phone) | how long a model stays loaded between messages |

## Keeping it alive

Android likes to stop background programs, and a model server is exactly that. Try these in this order (safest first):

1. **Keep Termux open** while you chat, and take a **wake lock**: `termux-wake-lock` (needs `pkg install termux-api` plus the Termux:API
   app), or tap *Acquire wakelock* in the Termux notification. Release with `termux-wake-unlock`. The installer offers this.
2. **Battery optimisation**: Settings > Apps > Termux > Battery > *Unrestricted* (wording differs per phone brand; some brands also
   have their own "app launch" or "protected apps" lists).
3. **Phantom process killer (Android 12 and newer)**: Android may kill the child processes of apps, such as the Ollama server, even
   while Termux is open. **UNVERIFIED on your device.** On Android 14 and newer, Developer options has *Disable child process
   restrictions*: enable it. On Android 12 to 13 the setting is only reachable from a PC with `adb` (a system setting; read about it
   before you change it):
   `adb shell settings put global settings_enable_monitor_phantom_procs false`.
   `nightage doctor` repeats this hint when it finds a local Ollama.
4. If Android still kills it, the next `nightage` run notices the dead server and starts a new one by itself.

If the whole Termux session is killed (swipe away, memory pressure, power button on some phones), nothing can run any more: no program
can clean up after a `SIGKILL`. Nightage handles `SIGHUP` (closing the terminal), `SIGTERM`, Ctrl+C and normal exits: it unloads the
models and stops the server it started. After a hard kill, the leftover server is found by the next `nightage` run (the pid file is
kept) and reused or stopped then.

## Freeing the memory

- Leaving the chat normally (`/quit`, Ctrl+C, closing the terminal) unloads the models and stops the Ollama Nightage started.
- Press Ctrl+C a second time during that cleanup to skip the waiting: the server is stopped at once.
- `nightage ollama stop` stops it by hand; `ollama ps` (or `nightage ollama status`) shows what is still loaded.
- Smaller `ollama.keep_alive` means the model leaves memory sooner between messages (the next message loads it again).

## How fast will it be?

**UNVERIFIED expectations** for a recent phone with 4 to 8 CPU cores (no GPU use; Ollama on Termux runs on the CPU or through Vulkan,
which was not tested here). Real numbers depend on the chip, cooling and what else runs.

| Model | Words appear at about | Notes |
|---|---|---|
| `qwen3.5:0.8b` | 10 to 20 tokens/s | fast, but simple answers; tools are hit and miss |
| `qwen3.5:2b` | 5 to 10 tokens/s | a fair balance |
| `qwen3.5:4b` | 2 to 5 tokens/s | the recommended tier with 6+ GB; a long answer takes a minute or more |
| `qwen3.5:9b` | under 2 tokens/s | needs ~7 GB free; expect very slow replies and throttling |

The first message after the model is loaded needs extra seconds (it reads the whole prompt first). A long conversation gets slower: the
phone profile keeps the context small on purpose. A phone that gets hot slows itself down.

## Troubleshooting

- **`ollama: command not found`** - run `pkg install ollama` (only 64-bit ARM and x86_64 phones have the package), then
  `nightage ollama start`. `nightage doctor` says the same.
- **The server does not start** - `nightage ollama status` shows the log path (`~/.local/state/nightage/ollama/server.log`). Look at the
  last lines. Port 11434 taken by another app: stop that one, or set `ollama.host: 127.0.0.1:11500`.
- **A package fails to install on a very new Python (3.14)** - the installer only needs pure-Python packages, but one of them may have
  no ready-made build for Android yet and tries to compile (needs `pkg install clang make`, which can take a while). Look at the last
  lines of the failure: if it names a compiler, install `clang` and run the installer again; if a version is missing, `pkg install python-pip` and retry later or ask for help with the log attached.
- **Installer fallbacks (UNVERIFIED on a phone)** - the installer tries `pkg install` three times (mirror hiccups), and if pip cannot build a package it retries with ready-made builds only (`--only-binary :all:`), then with Termux's own `python-yaml` and a virtual environment that can see system packages.
- **`pkg` cannot download (mirror errors, 404)** - `pkg update`; or `termux-change-repo` and pick another mirror; then run the installer again.
- **No access to Downloads/Documents** - `termux-setup-storage` (once), then the files are under `~/storage`.
- **It stops when the screen turns off** - see [Keeping it alive](#keeping-it-alive): wake lock, battery optimisation, child process restrictions.
- **Out of memory / Termux closes / "Killed"** - the model is too big for the free memory. Use a smaller one: `nightage pull qwen3.5:2b`,
  then choose it in `nightage settings`; close other apps; stay on `--profile phone` (it is the default after the installer).
- **Replies are very slow** - a smaller model, the phone profile (`nightage --profile phone`), a shorter conversation (start a new chat now and then), and keep the phone cool and charging. If tools are not needed,
  a plain chat is cheaper than a coding session.
- **Garbled symbols or lines** - `NIGHTAGE_PLAIN=1 nightage` uses plain characters. The installer does the same on its own when the
  locale is not UTF-8.
- **The download stopped** - run the same `nightage pull <model>` again: Ollama keeps the finished parts.

## Uninstall

```sh
nightage ollama stop                   # if it is running
rm -f "$PREFIX/bin/nightage"           # the command
rm -rf ~/.nightage-venv                # the Python environment
rm -rf <the project folder>            # also holds config.yaml and data/ (your memory and chats): copy them first if you want them
pkg uninstall ollama                   # optional: the Ollama program; its models are in ~/.ollama (delete it to free the space)
rm -rf ~/.local/state/nightage         # logs and the server's pid file
```

More: [PORTABILITY.md](PORTABILITY.md), [INSTALL.md](INSTALL.md), [DOCTOR.md](DOCTOR.md).
