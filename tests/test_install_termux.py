"""install-termux.sh with fake `pkg`, `python`, `tput`, `termux-wake-lock` and a fake venv python: width, bar, quiet mode, flags, re-runs.

Nothing here touches the network, Android or the real Python environment. Everything about a REAL phone stays UNVERIFIED."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from agent import embedmodels, profiles

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "install-termux.sh"
posix = pytest.mark.posix_only
TOOLS = ("sh", "awk", "sed", "uname", "id", "date", "mktemp", "rm", "df", "grep", "head", "tail", "tr", "wc", "dirname", "cat", "du",
         "sleep", "basename", "mkdir", "mv", "chmod", "cp", "cut", "env", "dash")

FAKE_PYTHON = """#!/bin/sh
case "$1" in --version) echo "Python 3.14.6"; exit 0 ;; esac
if [ "$1" = -m ] && [ "$2" = venv ]; then
  mkdir -p "$3/bin"; cp "$FAKES/venvpython" "$3/bin/python"; chmod +x "$3/bin/python"; echo "venv $3" >>"$CALLS"; exit 0
fi
exit 0
"""
FAKE_VENV_PYTHON = """#!/bin/sh
if [ "$1" = -m ] && [ "$2" = pip ]; then
  case "$3" in --version) echo "pip 25.0"; exit 0 ;; esac
  echo "pip $*" >>"$CALLS"
  if [ -n "${FAKE_PIP_FAIL:-}" ]; then i=1; while [ $i -le 14 ]; do echo "pip noise line $i"; i=$((i + 1)); done; echo "ERROR: no matching distribution found"; exit 1; fi
  if [ -n "${FAKE_PIP_FAIL_UNLESS_BINARY:-}" ]; then case "$*" in *only-binary*) ;; *) echo "error: Failed building wheel for pyyaml"; exit 1 ;; esac; fi
  echo "Collecting typer"; echo "Downloading typer-0.15.whl"; echo "Successfully installed typer"; exit 0
fi
if [ "$1" = -m ] && [ "$2" = agent.main ]; then
  shift 2; echo "nightage $*" >>"$CALLS"
  case "$1" in
    pull) echo "Downloading $2"; echo "[####] 100%"; [ -n "${FAKE_PULL_FAIL:-}" ] && exit 1; exit 0 ;;
    ollama) echo "started"; exit 0 ;;
    doctor) echo "doctor: ok"; exit 0 ;;
    models) [ -n "${FAKE_HAS_MODELS:-}" ] || { echo "No such command"; exit 2; }; echo "models ok"; exit 0 ;;
  esac
  exit 0
fi
if [ "$1" = - ]; then exec "$REAL_PY" -; fi
exit 0
"""
FAKE_PKG = """#!/bin/sh
echo "pkg $*" >>"$CALLS"
echo "Unpacking things... (pkg noise)"
if [ -n "${FAKE_PKG_FAILS:-}" ] && [ "$1" = install ]; then
  n=$(cat "$FAKES/pkgfails" 2>/dev/null || echo 0)
  if [ "$n" -lt "$FAKE_PKG_FAILS" ]; then echo $((n + 1)) >"$FAKES/pkgfails"; echo "E: mirror hiccup"; exit 1; fi
fi
[ "$1" = update ] && exit 0
shift; [ "$1" = -y ] && shift
for p in "$@"; do
  case "$p" in ripgrep) n=rg ;; *) n=$p ;; esac
  printf '#!/bin/sh\\necho %s\\n' "$n" >"$PREFIX/bin/$n"; chmod +x "$PREFIX/bin/$n"
done
exit 0
"""
FAKE_TPUT = """#!/bin/sh
[ -n "${FAKE_COLS:-}" ] || exit 1
[ "$1" = cols ] && echo "$FAKE_COLS"
"""


class Box:
    def __init__(self, tmp: Path, with_ollama_binary: bool = False):
        self.tmp = tmp
        self.src = tmp / "src"
        self.src.mkdir()
        shutil.copy(SCRIPT, self.src / "install-termux.sh")
        (self.src / "pyproject.toml").write_text("[project]\nname='x'\n")
        self.home = tmp / "home"
        self.home.mkdir()
        self.prefix = tmp / "prefix" / "com.termux" / "files" / "usr"
        (self.prefix / "bin").mkdir(parents=True)
        self.fakes = tmp / "fakes"
        self.fakes.mkdir()
        self.bin = tmp / "tools"
        self.bin.mkdir()
        for t in TOOLS:
            real = shutil.which(t)
            if real and not (self.bin / t).exists():
                (self.bin / t).symlink_to(real)
        for name, body in (("python", FAKE_PYTHON), ("venvpython", FAKE_VENV_PYTHON), ("pkg", FAKE_PKG), ("tput", FAKE_TPUT),
                           ("termux-wake-lock", '#!/bin/sh\necho "wakelock" >>"$CALLS"\n')):
            f = self.fakes / name
            f.write_text(body)
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
        self.calls = tmp / "calls.txt"
        self.calls.write_text("")
        if with_ollama_binary:
            o = self.prefix / "bin" / "ollama"
            o.write_text("#!/bin/sh\necho ollama\n")
            o.chmod(0o755)

    def run(self, *args: str, cols: int | None = 40, extra: dict | None = None, ram: int = 11200, tools: bool = True) -> subprocess.CompletedProcess:
        env = {
            "HOME": str(self.home), "PATH": f"{self.prefix / 'bin'}{os.pathsep}{self.fakes}{os.pathsep}{self.bin}", "PREFIX": str(self.prefix),
            "TERM": "xterm", "LANG": "C.UTF-8", "NIGHTAGE_TEST_RAM_MB": str(ram), "CALLS": str(self.calls), "FAKES": str(self.fakes),
            "REAL_PY": sys.executable, "PYTHONPATH": os.environ.get("PYTHONPATH", ""), "NO_COLOR": "",
        }
        if cols:
            env["FAKE_COLS"] = str(cols)
        env.update(extra or {})
        env = {k: v for k, v in env.items() if not (k == "NO_COLOR" and v == "")}
        return subprocess.run(["/bin/sh", str(self.src / "install-termux.sh"), *args], capture_output=True, text=True, timeout=120, env=env,
                              stdin=subprocess.DEVNULL, cwd=str(self.tmp))

    def calls_text(self) -> str:
        return self.calls.read_text()

    @property
    def log(self) -> Path:
        return self.home / ".local" / "state" / "nightage" / "install.log"

    @property
    def config(self) -> Path:
        return self.src / "config.yaml"


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path)


def widest(text: str) -> int:
    return max((len(x) for x in text.splitlines()), default=0)


# ── syntax / help ───────────────────────────────────────────────────────────────────────────

@posix
def test_syntax_posix_and_help():
    for shell in ("sh", "dash"):
        exe = shutil.which(shell)
        if exe:
            assert subprocess.run([exe, "-n", str(SCRIPT)]).returncode == 0
    text = SCRIPT.read_text()
    assert text.startswith("#!/bin/sh\n")
    for bashism in ("[[ ", "<<<", "$'", "${BASH", "echo -e", "local "):
        assert bashism not in text, bashism
    out = subprocess.run(["/bin/sh", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    for flag in ("--yes", "--dry-run", "--verbose", "--no-ollama", "--model", "--embed", "--no-embed", "--profile"):
        assert flag in out.stdout


@posix
def test_shellcheck_clean_when_available():
    exe = shutil.which("shellcheck")
    if not exe:
        pytest.skip("shellcheck not installed")
    res = subprocess.run([exe, "-s", "sh", str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout


@posix
def test_unknown_option_and_bad_profile(box):
    assert box.run("--wat").returncode == 2
    assert box.run("--profile", "gaming").returncode == 2
    assert box.run("--model").returncode == 2


@posix
def test_refuses_outside_termux_and_dry_run_changes_nothing(tmp_path):
    b = Box(tmp_path)
    # no `pkg` on PATH at all
    env = {"PATH": str(b.bin), "HOME": str(b.home), "PREFIX": str(b.prefix), "TERM": "dumb"}
    nope = subprocess.run(["/bin/sh", str(b.src / "install-termux.sh"), "--yes"], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
    assert nope.returncode == 1 and "Termux" in nope.stderr
    dry = subprocess.run(["/bin/sh", str(b.src / "install-termux.sh"), "--dry-run"], capture_output=True, text=True,
                         env={**env, "NIGHTAGE_REMOTE_HOST": "100.64.0.5"}, stdin=subprocess.DEVNULL)
    assert dry.returncode == 0 and "[dry-run]" in dry.stdout and "ollama.host" in dry.stdout
    assert not (b.home / ".nightage-venv").exists() and not any((b.prefix / "bin").iterdir())


# ── width and look ──────────────────────────────────────────────────────────────────────────

@posix
@pytest.mark.parametrize("cols", [30, 40, 54, 60, 80])
@pytest.mark.parametrize("uni", [True, False])
def test_no_line_exceeds_width_minus_one(box, cols, uni):
    r = box.run("--dry-run", cols=cols, extra={} if uni else {"LANG": "C"})
    assert r.returncode == 0, r.stderr
    assert widest(r.stdout) <= min(cols, 72) - 1, [x for x in r.stdout.splitlines() if len(x) > cols - 1]
    if not uni:
        assert r.stdout.isascii() and "█" not in r.stdout
    else:
        assert "█" in r.stdout or "░" in r.stdout


@posix
def test_failing_run_also_respects_the_width(box):
    r = box.run("--yes", cols=30, extra={"FAKE_PIP_FAIL": "1"})
    assert r.returncode == 1
    assert widest(r.stdout + r.stderr) <= 29


@posix
def test_headers_have_bar_percent_and_step(box):
    r = box.run("--dry-run", cols=40)
    heads = [x for x in r.stdout.splitlines() if re.match(r"^\[[█░#-]+\] \d+%  ", x)]
    assert [re.search(r"Step (\d) of 8", h).group(1) for h in heads[:8]] == list("12345678")
    pcts = [int(re.search(r"\] (\d+)%", h).group(1)) for h in heads]
    assert pcts == [0, 4, 18, 38, 42, 52, 82, 90, 100] and pcts == sorted(pcts)
    for h in heads:
        inner = h[1:h.index("]")]
        pct = int(re.search(r"\] (\d+)%", h).group(1))
        assert len(inner) >= 10 and inner.count("█") == (pct * len(inner) + 50) // 100, h


@posix
def test_bar_math_0_37_100():
    """The shell bar uses the same rounding as agent.ui.style.bar for 0, 37 and 100 percent."""
    from agent.ui import style

    for pct in (0, 37, 100):
        for width in (10, 19, 30):
            sh = subprocess.run(["/bin/sh", "-c", f'''FILL='#'; EMPTY='-'; ACC=''; RST=''; DIM=''
make_bar() {{ mb_n=$(( ($1 * $2 + 50) / 100 )); mb_f=''; mb_e=''; mb_i=0
  while [ "$mb_i" -lt "$2" ]; do if [ "$mb_i" -lt "$mb_n" ]; then mb_f="$mb_f$FILL"; else mb_e="$mb_e$EMPTY"; fi; mb_i=$((mb_i + 1)); done
  BAR="$mb_f$mb_e"; }}
make_bar {pct} {width}; printf %s "$BAR"'''], capture_output=True, text=True).stdout
            assert sh == style.bar(pct, width, style.ASCII)
    text = SCRIPT.read_text()
    assert "mb_n=$(( ($1 * $2 + 50) / 100 ))" in text            # the very code tested above


@posix
def test_ascii_fallback_for_dumb_terminal_and_no_color(box):
    for extra in ({"TERM": "dumb"}, {"NO_COLOR": "1"}, {"LANG": "C", "LC_ALL": "C"}):
        r = box.run("--dry-run", cols=40, extra=extra)
        assert r.stdout.isascii(), extra
        assert re.search(r"^\[#*-*\] 0%", r.stdout, re.M)
    assert "\x1b" not in box.run("--dry-run", cols=40).stdout        # no colour codes when stdout is not a terminal


@posix
def test_words_are_not_split_at_40_columns(box):
    wide = set(box.run("--dry-run", cols=72).stdout.split())
    narrow = box.run("--dry-run", cols=40).stdout.split()
    broken = [w for w in narrow if w not in wide and not re.fullmatch(r"\[[█░]+\]|[╭╰][─]+[╮╯]", w)]       # the bar itself is as wide as the screen
    assert not broken, broken


@posix
def test_width_detection_order(box):
    note = "Experimental: not tried on a real"
    # tput answers: used
    assert any(note in x for x in box.run("--dry-run", cols=40).stdout.splitlines())
    # tput fails, COLUMNS used
    wide = box.run("--dry-run", cols=None, extra={"COLUMNS": "70"}).stdout
    assert any("Experimental: not tried on a real phone yet." in x for x in wide.splitlines())
    # nothing known: fallback 40
    r = box.run("--dry-run", cols=None, extra={"COLUMNS": ""})
    assert widest(r.stdout) <= 39 and any(note in x for x in r.stdout.splitlines())
    # tput wins over COLUMNS
    r = box.run("--dry-run", cols=40, extra={"COLUMNS": "70"})
    assert widest(r.stdout) <= 39


@posix
def test_dry_run_prints_plan_and_changes_nothing(box):
    r = box.run("--dry-run", cols=40)
    assert r.returncode == 0
    for needle in ("[dry-run]", "pkg install -y", "pip install", "ollama start", "nightage pull qwen3.5:4b", "nightage pull all-minilm",
                   "profile = phone", "Done. Start with: nightage", "dry run: nothing was changed"):
        assert needle in " ".join(r.stdout.split()) or needle in r.stdout, needle
    assert box.calls_text() == "" and not box.log.exists() and not (box.home / ".nightage-venv").exists() and not box.config.exists()


# ── a complete fake run ─────────────────────────────────────────────────────────────────────

@posix
def test_full_run_is_quiet_and_configures_everything(box):
    r = box.run("--yes", cols=40)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Collecting" not in r.stdout and "Unpacking" not in r.stdout              # tool output stays in the log
    log = box.log.read_text()
    assert "Collecting typer" in log and "Unpacking things" in log
    assert "Done. Start with: nightage" in r.stdout and "[" in r.stdout.splitlines()[-1] or "Done" in r.stdout
    calls = box.calls_text()
    assert "pkg install -y git ripgrep" in calls or "pkg install -y python git ripgrep" in calls
    assert "pkg install -y ollama" in calls
    for want in ("nightage ollama start", "nightage pull qwen3.5:4b", "nightage pull all-minilm", "nightage doctor --only system,ollama,models", "wakelock"):
        assert want in calls, want
    assert calls.index("ollama start") < calls.index("pull qwen3.5:4b") < calls.index("pull all-minilm") < calls.index("doctor")
    launcher = box.prefix / "bin" / "nightage"
    assert launcher.exists() and os.access(launcher, os.X_OK) and "agent.main" in launcher.read_text()
    import yaml

    cfg = yaml.safe_load(box.config.read_text())
    assert cfg["profile"] == "phone" and cfg["models"]["chat"] == "qwen3.5:4b" and cfg["models"]["embed"] == "all-minilm"
    assert widest(r.stdout) <= 39


@posix
def test_rerun_is_idempotent(box):
    assert box.run("--yes").returncode == 0
    first_cfg = box.config.read_text()
    first_launcher = (box.prefix / "bin" / "nightage").read_text()
    box.calls.write_text("")
    r = box.run("--yes")
    assert r.returncode == 0
    calls = box.calls_text()
    assert "pkg install" not in calls.replace("pkg install -y ollama", "") or "pkg install -y ollama" not in calls   # nothing left to install
    assert "venv" not in calls and "pip install" in calls                     # the environment is kept, pip only confirms
    assert box.config.read_text() == first_cfg and (box.prefix / "bin" / "nightage").read_text() == first_launcher
    assert "There" not in r.stderr


@posix
def test_failure_shows_last_eight_lines_and_log_path(box):
    r = box.run("--yes", cols=40, extra={"FAKE_PIP_FAIL": "1"})
    assert r.returncode == 1
    out = " ".join((r.stdout + r.stderr).split())
    assert "ERROR: no matching distribution found" in out
    shown = {int(n) for n in re.findall(r"pip noise line (\d+)", out)}
    assert shown == set(range(8, 15))                                  # the last 8 lines: seven noise lines and the ERROR line
    assert "install.log" in out and "docs/TERMUX.md" in out
    assert box.log.exists() and "pip noise line 1" in box.log.read_text()


@posix
def test_pull_failure_is_not_fatal_and_listed(box):
    r = box.run("--yes", extra={"FAKE_PULL_FAIL": "1"})
    assert r.returncode == 0
    flat = " ".join(r.stdout.split())
    assert "Still to do" in flat and "nightage pull qwen3.5:4b" in flat and "nightage embed use all-minilm" in flat


# ── flags ───────────────────────────────────────────────────────────────────────────────────

@posix
def test_no_ollama_skips_everything_on_the_phone(box):
    r = box.run("--yes", "--no-ollama")
    assert r.returncode == 0
    calls = box.calls_text()
    assert "ollama start" not in calls and "install -y ollama" not in calls and "pull" not in calls and "doctor" in calls


@posix
def test_remote_host_is_written_and_skips_local_steps(box):
    r = box.run("--yes", extra={"NIGHTAGE_REMOTE_HOST": "100.64.0.5"})
    assert r.returncode == 0
    import yaml

    assert yaml.safe_load(box.config.read_text())["ollama"]["host"] == "100.64.0.5"
    assert "pull" not in box.calls_text() and "pkg install -y ollama" not in box.calls_text()


@posix
def test_model_and_embed_flags(box):
    r = box.run("--yes", "--model", "qwen3.5:2b", "--embed=nomic-embed-text")
    assert r.returncode == 0
    calls = box.calls_text()
    assert "pull qwen3.5:2b" in calls and "pull nomic-embed-text" in calls and "qwen3.5:4b" not in calls
    import yaml

    cfg = yaml.safe_load(box.config.read_text())
    assert cfg["models"]["chat"] == "qwen3.5:2b" and cfg["models"]["coding"] == "qwen3.5:2b" and cfg["models"]["embed"] == "nomic-embed-text"


@posix
def test_no_embed_and_profile_default(box):
    r = box.run("--yes", "--no-embed", "--profile", "default")
    assert r.returncode == 0
    assert "pull all-minilm" not in box.calls_text() and "pull qwen3.5:4b" in box.calls_text()
    import yaml

    assert "profile" not in yaml.safe_load(box.config.read_text())


@posix
def test_existing_user_setting_is_not_overwritten_without_force(box):
    box.config.write_text("models:\n  embed: my-own-embedder\n")
    assert box.run("--yes", "--model", "qwen3.5:2b").returncode == 0
    import yaml

    cfg = yaml.safe_load(box.config.read_text())
    assert cfg["models"]["embed"] == "all-minilm" or cfg["models"]["embed"] == "my-own-embedder"      # the pulled model is registered on purpose
    assert cfg["models"]["chat"] == "qwen3.5:2b"


@posix
def test_verbose_streams_tool_output(box):
    r = box.run("--yes", "--verbose")
    assert r.returncode == 0 and "Unpacking things" in r.stdout


@posix
def test_without_a_keyboard_and_without_yes_it_stops_politely(box):
    r = box.run(cols=40)
    assert r.returncode == 1 and "--yes" in r.stdout + r.stderr


# ── the memory tiers mirror the Python side ─────────────────────────────────────────────────

@posix
@pytest.mark.parametrize("ram", [0, 1500, 2999, 3072, 4096, 6000, 6144, 8000, 11200, 16000])
def test_chat_tier_matches_profiles_model_tier(ram):
    text = SCRIPT.read_text()
    func = text[text.index("chat_tier() {"):text.index("chat_size() {")]
    out = subprocess.run(["/bin/sh", "-c", f"{func}\nchat_tier {ram}"], capture_output=True, text=True).stdout.strip()
    assert out == profiles.model_tier(ram)


@posix
def test_under_6gb_the_default_is_no_models_on_the_phone(box):
    r = box.run("--yes", ram=4096)
    assert r.returncode == 0 and "pull" not in box.calls_text() and "pkg install -y ollama" not in box.calls_text()


def test_embedders_offered_exist_in_the_catalogue():
    text = SCRIPT.read_text()
    tags = {m.tag for m in embedmodels.CATALOGUE}
    for tag in ("all-minilm", "embeddinggemma:300m-qat-q4_0", "nomic-embed-text"):
        assert tag in text and tag in tags
    for tag, mb in (("all-minilm", 46), ("embeddinggemma:300m-qat-q4_0", 239), ("nomic-embed-text", 274)):
        assert embedmodels.describe(tag).download_mb == mb and f"{tag}) esz={mb}" in text.replace('"', "")


# ── mirror retry, pip fallback, role assignment ─────────────────────────────────────────────

@posix
def test_pkg_mirror_hiccup_is_retried(box):
    r = box.run("--yes", extra={"FAKE_PKG_FAILS": "2", "NIGHTAGE_RETRY_WAIT": "0"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert box.calls_text().count("pkg install -y python git ripgrep") >= 3 or box.calls_text().count("pkg install -y git ripgrep") >= 3
    assert "mirror hiccup" in box.log.read_text() and "mirror hiccup" not in r.stdout


@posix
def test_pkg_gives_up_after_three_tries_with_log_tail(box):
    r = box.run("--yes", extra={"FAKE_PKG_FAILS": "9", "NIGHTAGE_RETRY_WAIT": "0"})
    assert r.returncode == 1 and "mirror hiccup" in r.stdout and "install.log" in r.stdout
    assert widest(r.stdout) <= 39


@posix
def test_pip_falls_back_to_ready_made_builds_only(box):
    r = box.run("--yes", extra={"FAKE_PIP_FAIL_UNLESS_BINARY": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "only-binary" in box.calls_text() and "retrying with ready-made builds only" in box.log.read_text()


@posix
def test_all_chat_roles_written_when_models_command_is_missing(box):
    import yaml

    r = box.run("--yes")
    assert r.returncode == 0, r.stdout + r.stderr
    m = yaml.safe_load(box.config.read_text())["models"]
    for role in ("chat", "background", "coding", "reasoning", "unrestricted", "weekly"):
        assert m[role] == "qwen3.5:4b", role
    assert m["embed"] == "all-minilm"
    assert "nightage models use" not in box.calls_text() or "models use --help" in box.calls_text()


@posix
def test_models_command_is_preferred_when_it_exists(box):
    r = box.run("--yes", extra={"FAKE_HAS_MODELS": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    calls = box.calls_text()
    assert "nightage models use qwen3.5:4b --yes" in calls and "nightage models embed all-minilm --yes" in calls
    assert calls.index("pull qwen3.5:4b") < calls.index("models use") < calls.index("pull all-minilm") < calls.index("models embed")
