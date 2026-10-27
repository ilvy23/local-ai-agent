# Installing Nightage

**Short version.** Open a terminal, paste ONE line, press Enter, answer the questions (just press Enter to accept the safe answer).
Then type `nightage`. You do not need to know Python, and you do not need an administrator password.

Replace `OWNER/REPO` below with this project's GitHub page (the address of the repository you are reading this in).

| You use | Paste this |
|---|---|
| **Windows** (in "Windows PowerShell") | `irm https://raw.githubusercontent.com/OWNER/REPO/main/install.ps1 \| iex` |
| **macOS** (in "Terminal") | `curl -LsSf https://raw.githubusercontent.com/OWNER/REPO/main/install.sh \| sh` |
| **Linux** (any terminal) | `curl -LsSf https://raw.githubusercontent.com/OWNER/REPO/main/install.sh \| sh` |
| **Android** (Termux) | `curl -LsSf https://raw.githubusercontent.com/OWNER/REPO/main/install.sh \| sh` (then it hands over to `install-termux.sh`) |

> Windows, macOS and Termux support has not been tried on real machines by the author (**UNVERIFIED**; only tests and CI have run it).
> If something goes wrong, the troubleshooting section below and the log file the installer prints will help; please report it.

## What you will see (Windows example)

```
 Nightage installer  (a local AI in your terminal; nothing leaves your computer)

[1/7] What will happen
   | This guided installer will, step by step and only after asking you:
   |   - check your computer (read only)
   |   - install uv (official tool from astral.sh, into your user folder) - it brings its own Python
   |   - install Nightage in its own private environment (managed by uv)
   |   - help you install Ollama, which runs the AI model on your computer
   |   - download ONE AI model that fits your computer (several GB; you see the size first)
   | It never needs administrator rights, never edits system settings, and prints every command before running it.
   ? Ready to start? [Y/n]

[2/7] Looking at your computer
   + System: Microsoft Windows 11 Home, AMD64
   + Memory (RAM): 7.7 GB
   + Graphics: NVIDIA GeForce GTX 1650 ...
[3/7] Python, the easy way (uv)
   | It will download this script over HTTPS and run it as you (no admin):  https://astral.sh/uv/install.ps1
   ? Install uv now? [Y/n]
[4/7] Installing Nightage          ... ? Install Nightage now? [Y/n]
[5/7] Ollama (the engine that runs the AI)   ... ? Install Ollama now? [Y/n]
[6/7] The AI model
   | For this computer (about 3.8 GB usable for a model, from half of the RAM) the best fit is: qwen3.5:4b  (download about 3.9 GB)
   ? Download qwen3.5:4b now (one time)? [Y/n]
[7/7] Checking everything
 All done.
     nightage          start here: the home screen
     nightage demo     see what it looks like, no model needed
     nightage chat     start talking
```

Every step is a question with a default in brackets; Enter picks the capital letter. Nothing is changed until you say yes, and the exact
command is printed before it runs.

## "I do not know Python" path

You do not need to. The installer uses a small tool called **uv** that downloads its own private Python for Nightage. It does not touch
any Python you may already have, and Nightage cannot break other programs. If the word "terminal" is new to you:

* **Windows:** press the Start key, type `PowerShell`, press Enter. A blue window opens. Paste the line (right-click pastes in that window), press Enter.
* **macOS:** press Cmd+Space, type `Terminal`, press Enter. Paste the line (Cmd+V), press Enter.
* **Linux:** open your terminal program. Paste (Ctrl+Shift+V), press Enter.

When it is done, **close the window and open a new one**, then type `nightage` and press Enter.

## Prefer to look first? (download, read, then run)

Piping a script into a shell is convenient, but you are allowed to read it first. It is short and plain text:

```sh
curl -LsSf https://raw.githubusercontent.com/OWNER/REPO/main/install.sh -o install.sh
less install.sh            # q to quit
sh install.sh              # run it
sh install.sh --dry-run    # or: show everything it WOULD do, change nothing
```

```powershell
irm https://raw.githubusercontent.com/OWNER/REPO/main/install.ps1 -OutFile install.ps1
notepad install.ps1
powershell -ExecutionPolicy Bypass -File install.ps1            # works even when scripts are blocked
powershell -ExecutionPolicy Bypass -File install.ps1 -DryRun    # show everything, change nothing
```

Options with the one-liner on Windows (the plain one-liner cannot take options):
`& ([scriptblock]::Create((irm https://raw.githubusercontent.com/OWNER/REPO/main/install.ps1))) -Yes`

Already have the project folder (downloaded zip or `git clone`)? Run `sh install.sh` / `install.ps1` from inside it: it installs from that
folder, no address needed.

## Options

| `install.sh` | `install.ps1` | Meaning |
|---|---|---|
| `--yes` | `-Yes` | accept the safe default for every question (never uses sudo, never deletes your data) |
| `--dry-run` | `-DryRun` | print everything, change nothing |
| `--no-ollama` | `-NoOllama` | do not touch Ollama (it runs on another machine, or you manage it yourself) |
| `--no-model` | `-NoModel` | do not offer a model download |
| `--model NAME` | `-Model NAME` | use this model instead of the recommended one |
| `--with-ollama` | `-WithOllama` | explicit yes to the official Ollama installer (Linux: this needs sudo) |
| `--source X` | `-Source X` | install from a folder, an `https://` archive (`.tar.gz`/`.zip`), or a `git+https://` URL. Env: `NIGHTAGE_REPO=https://github.com/OWNER/REPO` |
| `--uninstall` | `-Uninstall` | remove Nightage; asks before touching your data |
| `--force` | `-Force` | allow running as root / Administrator (not recommended) |
| `--help` | `-Help` | the list |

Environment: `NIGHTAGE_PYTHON` (Python version uv should use, default 3.12), `NIGHTAGE_INSTALL_LOG` (log file), `NO_COLOR` (no colours).

Installing a specific release instead of the latest: `--source https://github.com/OWNER/REPO/archive/refs/tags/vX.Y.Z.tar.gz`.

## Per system

### Windows (PowerShell 5.1 or 7, Windows 10/11)

1. Open **Windows PowerShell** (normal window, not "as administrator").
2. Paste the one-liner. The installer installs **uv** (user folder `%USERPROFILE%\.local\bin`), then Nightage as a uv tool, then helps with
   **Ollama** (`winget install Ollama.Ollama` when winget exists, otherwise the official `OllamaSetup.exe` is downloaded from `ollama.com` and
   its digital signature is checked before it runs).
3. Open a **new** PowerShell window and type `nightage`.

The terminal UI looks best in **Windows Terminal** (Windows 11 has it). In the old console window, colours and line drawing are reduced
automatically. The shell-based **coding agent** (`nightage code`) runs commands through `bash`; on Windows it needs **WSL** (recommended) or
Git for Windows. Chat, memory, demo and doctor work without it. Details and status: [PORTABILITY.md](PORTABILITY.md).

### macOS

The one-liner works in Terminal. Ollama: if you have Homebrew the installer offers `brew install ollama`; otherwise it points you to the
official app at https://ollama.com/download. Apple Silicon Macs share memory between CPU and GPU, so the model is chosen by RAM.

### Linux

Per-user install, no sudo. The one thing that needs sudo is Ollama's own official installer (it creates a system service). The installer
explains that and only runs it if you answer **yes** to that specific question (plain `--yes` never does; `--with-ollama` is the explicit yes).
You can also install Ollama yourself from https://ollama.com/download and run the installer again.

### Termux (Android)

`install.sh` detects Termux and runs `install-termux.sh` (lighter: no compiled packages). Read [TERMUX.md](TERMUX.md); it is **UNVERIFIED**
on real phones. A phone usually does better talking to Ollama on your PC (`ollama.host` in the settings).

## Troubleshooting

**`nightage` is "not recognized" / "command not found".** The installer adds the tool folder to YOUR user PATH, but windows that were already
open do not see it. Close the window and open a new one. Still nothing? Run `uv tool update-shell` and open a new window, or run it by full
path: Windows `%USERPROFILE%\.local\bin\nightage.exe`, others `~/.local/bin/nightage`.

**Windows: "running scripts is disabled on this system" (execution policy).** The one-liner (`irm ... | iex`) is not affected, because
nothing is saved as a script file. For a downloaded file use `powershell -ExecutionPolicy Bypass -File install.ps1`. That only applies to that
one run; it does not change your settings.

**Windows: SmartScreen "Windows protected your PC".** This appears for the Ollama installer or a downloaded `.exe` that is new to Microsoft.
The installer checks Ollama's digital signature itself and shows who signed it. Only click "Run anyway" if the publisher shown is
Ollama / uv and the file came from the address the installer printed.

**Antivirus blocks something.** Some security programs block scripts that download programs. Both scripts are plain text you can read; if your
antivirus insists, install uv and Ollama by hand from their official pages and run the installer again (it skips what is already there).

**Behind a proxy / company network.** Set the proxy for your window first, then run the installer again.
Windows: `$env:HTTPS_PROXY = "http://proxy.example:8080"`. macOS/Linux: `export HTTPS_PROXY=http://proxy.example:8080`.
Problems with certificates usually mean the company inspects traffic: ask your IT department for the certificate bundle.

**No admin rights.** Fine: nothing here needs them. Ollama's Windows installer is per-user. On Linux, Ollama's official installer needs sudo;
without it follow the "no sudo" manual steps the installer prints, or use an Ollama running on another machine (`--no-ollama` and set `ollama.host`).

**"Ollama not found" / the model cannot be downloaded.** Install Ollama (step 5), then start it (Windows: it starts by itself and sits in the tray;
others: `ollama serve`), then run the installer again. Check with `ollama list`.

**The model is too big for my RAM.** The installer picks by your memory (half of RAM, or 90% of GPU memory). If it is still too slow or the
computer freezes, choose a smaller one: `ollama pull qwen3.5:2b` then `nightage settings set chat_model qwen3.5:2b`.
`nightage doctor --deep` measures speed and recommends a model.

**Unicode boxes look broken / question marks.** Use Windows Terminal, or set `NO_COLOR=1` (also switches to plain ASCII drawing).

**Something else.** Run `nightage doctor` (it explains problems in plain words), and for a bug report attach the install log (path printed at
the end; Windows `%LOCALAPPDATA%\nightage\install.log`, others `~/.local/state/nightage/install.log`) and `nightage doctor --report`
(anonymised).

## Updating

```
nightage update --check     # look only
nightage update             # shows what is new and the exact commands, asks, updates
```

Or run the installer again: it is safe to repeat and acts as repair / upgrade. Details: [UPDATING.md](UPDATING.md).

## Uninstalling

```sh
sh install.sh --uninstall                                          # Linux / macOS
powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall    # Windows
```

It removes the program and asks before touching your data (settings, memory database, history: the default is to keep them; to delete you
must type the word `delete`). By hand: `uv tool uninstall agent`. Your data lives in `~/.local/share/nightage` and `~/.config/agent`
(Windows: `%LOCALAPPDATA%\nightage`, `%LOCALAPPDATA%\agent`, `%APPDATA%\agent`). The installer never removes uv, Python or Ollama: remove uv with
`winget uninstall astral-sh.uv` or by deleting `uv`/`uvx` from `~/.local/bin`, and the Pythons it downloaded from the folder `uv python dir` prints;
remove Ollama like any other program (Windows: Settings > Apps) and its models with `ollama rm <name>`.

## What the installer does (the security explanation)

* **Per user.** No admin, no root, no sudo (it refuses to run as root/Administrator unless you add `--force`). It never edits system files or
  system settings. The only configuration it may change is your own shell startup file or your user PATH, to make `nightage` findable, and
  only after asking (`uv tool update-shell`).
* **Everything is shown first.** Every command is printed before it runs; `--dry-run` shows the whole plan without doing anything.
* **Official sources over HTTPS only.** `https://astral.sh/uv/install.sh` / `install.ps1` (uv), `https://ollama.com/...` (Ollama), the model from
  Ollama's own library, and Nightage from the project's GitHub page. Anything else is refused.
* **Downloaded installers are saved, not piped.** The uv and Ollama scripts are downloaded to a temporary file, their size and SHA-256 are
  printed, you can read them before they run, and the temporary folder is deleted afterwards. uv's installer verifies the program it fetches
  against checksums published with each release. Ollama publishes no checksum for its script/installer, so on Windows the digital
  signature of `OllamaSetup.exe` is verified instead (and an invalid one stops the install unless you insist).
* **Isolated.** Nightage lives in its own private Python environment (uv tool), so it cannot disturb other software and removing it is one command.
* **Sudo only on an explicit yes.** The one step that needs it (Ollama's Linux installer: it installs into `/usr/local` and creates a system
  service) is explained, defaults to **no**, and is skipped by `--yes`.
* **Repeatable.** Safe to run again; it only does what is missing and upgrades Nightage.
* **A log for bug reports.** Printed at the start and the end; it contains what was shown on screen, not your files or passwords.
* **Nothing phones home.** The installer sends nothing to the project; the only traffic is to the download hosts above.
