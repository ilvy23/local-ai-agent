# nightage doctor

`nightage doctor` looks at the machine it runs on and at your setup, and tells you what is wrong and how to fix it. It is modular:
every check lives in its own small function, checks are grouped by area, and checks for another operating system are skipped
automatically (a NixOS machine gets NixOS checks, Termux gets Termux checks, and so on).

```
nightage doctor [--only GROUP,...] [--skip GROUP,...] [--deep] [--fix] [--json] [--list]
                [--verbose] [--no-web] [--report] [--strict] [--no-fix]
```

| option | what it does |
|---|---|
| `--only a,b` / `--skip a,b` | run only / leave out groups or check ids (`gpu`, `disk.data`, `ollama`, ...). `--list` shows them. |
| `--list` | every check, its group, which systems it applies to, and flags (deep, network, needs Ollama). With `--json` as JSON. |
| `--deep` | also run the tests that **use the model** (a few tokens each): time to first token and tokens/s, an embedding call, a tool-call test, then a "recommended model for this machine" table with the exact `ollama pull` line. |
| `--fix` | for each finding that has a one-command fix: show the exact command, ask (default **no**), run it. See "What --fix may run". |
| `--json` | machine-readable output (`status`, `summary`, `meta`, `checks[]` with id, group, status, detail, hint, command, ms). |
| `--verbose` | check ids, timings, extra rows (for example the installed model list). |
| `--no-web` | skip every check that needs the internet (DNS, HTTPS, web search). |
| `--report` | print an anonymised markdown report you can paste into a GitHub issue (see below). |
| `--strict` | exit code 2 when there are only warnings. |
| `--no-fix` | leave out the fix hints. |

Exit codes: `0` fine (warnings allowed), `1` at least one problem, `2` only warnings with `--strict`, `3` `--report` refused because its
self-check found something that was not anonymised. Output works at 40, 60 and 100 columns and in plain ASCII (`NO_COLOR`, pipes).

## What is checked

Statuses: `ok`, `warn`, `fail`, `info` (a note), `skip` (with the reason: other system, no Ollama, `--no-web`, a dependency failed).

| group | checks (id) | systems |
|---|---|---|
| system | `platform`, `system.arch`, `profile` (config group) | all |
| | `linux.distro`, `linux.cgroup` (memory limit), `linux.units` (systemd user services ollama-vk / ollama / worker) | linux / systemd |
| | `nixos.version`, `nixos.nixld`, `nixos.ollama_service` | nixos |
| | `storage`, `termux.api`, `termux.wakelock`, `termux.battery` | termux |
| | `macos.version`, `macos.arch` (Rosetta), `macos.ollama_app` | macos |
| | `windows.version`, `windows.terminal`; `windows.wsl` (data on /mnt, GPU passthrough) | windows; wsl |
| python | `python`, `python.env`, `python.encoding` | all |
| packages | `packages`, `packages.versions`, `tools` (git, rg, uv, ollama) | all |
| ollama | `ollama`, `remote_latency`, `ollama.host` (local/remote, plain http on public address), `ollama.binary`, `ollama.version`, `ollama.vulkan` (second server :11436 when configured), `ollama.loaded` (GPU/CPU split) | all |
| models | `models` (chat / coding / embed installed? exact `ollama pull`), `context`, `models.installed`, `models.fit` (will it fit VRAM/RAM), `models.capabilities` (tool support from `ollama show`) | all |
| gpu | `gpu`, `gpu.nvidia` (driver, CUDA, VRAM), `gpu.amd` (amdgpu, ROCm, /dev/kfd access), `gpu.intel`, `gpu.vulkan` (vulkaninfo, software renderer), `gpu.metal` | per vendor / OS |
| memory | `memory.ram`, `memory.swap`, `memory.pressure`, `memory.headroom` (RAM left after the chat model), `memory.roundtrip` (saves and recalls a fact in a throwaway database; by meaning when the memory model works, else by keyword, with the exact reason) | all / linux |
| disk | `disk.data`, `disk.models` | all |
| network | `web`, `network.dns`, `network.https`, `network.proxy` | all (`--no-web` skips) |
| ports | `ports.ollama` (11434), `ports.vulkan` (11436), `ports.web` (8420) | all |
| terminal | `terminal`, `terminal.env` (TERM, ssh, tmux, screen), `terminal.clipboard` | all |
| filesystem | `data`, `filesystem.dirs`, `filesystem.webtoken` (mode 0600) | all |
| database | `database.integrity` (read-only), `database.schema`, `database.size` (WAL) | all |
| config | `config`, `config.keys` (typos), `config.models`, `config.safety` (extend-only lists, over-broad `safe_commands`) | all |
| security | `security.web_auth`, `security.web_bind`, `security.trusted_ips`, `security.secrets` (plaintext secrets, values never printed), `security.ollama_exposed` | all |
| plugins | `plugins.manifest` (each plugin's own check), `plugins.checks` (checks plugins added) | all |
| update | `update.available` (from the cache, no network) | all |
| deep | `deep.generate`, `deep.embed`, `deep.toolcall`, `deep.recommend` | `--deep` only |

Termux, Android and Windows checks were written from the platforms' documentation and have not been verified on real devices.

## What --fix may run

Only user-level commands, shown in full before anything runs, one question each, default no, never without a terminal:
`ollama pull <model>`, `chmod <owner-only mode> <file in your home folder or the data folder>`, `mkdir -p <folder under your home>`, and
writing one top-level key (for example `profile: phone`) into config.yaml (a `.bak` copy is kept). `sudo`, `su`, shells, `rm`,
`systemctl` and package managers are refused by the code (`agent/doctor/fixes.py`), not just by convention. Plugin checks can never
offer a fix. With `--json`, `--fix` is ignored.

## Sharing a report: `nightage doctor --report`

Prints a markdown report (versions, system, GPU names, every check result) with the user name, host name, home folder, IPv4/IPv6 and MAC
addresses, e-mail addresses, URL credentials, tailnet names and token-shaped strings replaced (`~`, `<host>`, `<ip>`, ...). It contains no
chat content and no config values. Then it **checks its own output** for the same things, and refuses to print (exit 3, and it says which
kind of thing was left, never the value) if anything slipped through. Read it once before posting anyway.

## Adding a check (15 lines)

Drop a file into `agent/doctor/checks/` (or `checks/system/` for one OS); it is found automatically.

```python
from agent.doctor.core import Context, check, ok, warn

@check("disk.tmp", "disk", "Space in /tmp", systems=("linux", "macos"), timeout=5)
def tmp_space(ctx: Context):
    d = ctx.p.disk("/tmp")                      # always go through ctx.p (the probes): tests replace it with a fake
    if d is None:
        return None                              # nothing to say: no row
    free_gb = d[1] / 1024**3
    if free_gb < 1:
        return warn(f"only {free_gb:.1f} GB free", "Clear /tmp or point TMPDIR elsewhere.")
    return ok(f"{free_gb:.1f} GB free")
```

`@check(id, group, title, systems=None, deep=False, needs_ollama=False, network=False, timeout=8, requires=(), core=False)`:
`systems` takes `linux macos windows termux android nixos wsl systemd` or a distro id; `needs_ollama` skips with a reason when the server is
down; `requires=("ollama",)` skips when that check failed; `network=True` runs in a parallel thread and honours `--no-web`; `deep=True`
only with `--deep`. A check returns a `Result` (`ok/warn/fail/info/skip(detail, hint)`), a list of them (give each its own `id=`), or
`None`. An exception or a timeout becomes a "could not run" warning. To offer a fix set `command=` and `fix=` from
`agent.doctor.fixes` (it only builds the allowed commands). Test with `tests/_doctorfake.py::FakeProbes`.

A **plugin** adds checks with `api.doctor_check("id", "Title", fn)`, where `fn()` returns a string, `(status, detail)` or
`(status, detail, hint)` (see [PLUGINS.md](PLUGINS.md)).

## Code map

`agent/doctor/core.py` (Result, registry, runner), `probes.py` (everything that touches the machine), `checks/*.py`, `fixes.py`,
`report.py`, `anonymise.py`, `cli.py` (`run(...)` returns the exit code; `register(app)` adds the typer command),
`legacy.py` (the older ok/warn/fail API that `agent/ui/doctor.py` re-exports for the home screen and onboarding).
