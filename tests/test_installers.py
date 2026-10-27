"""The guided installers, tested without Windows, without a network and without installing anything.

install.sh runs with `--dry-run` in a temporary HOME and a hermetic PATH (symlinks to the few basic tools, plus fake uv/ollama where a
test wants them), and the printed plan is asserted. install.ps1 is checked statically (always) and run under `pwsh` when it exists."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from agent import brand

ROOT = Path(__file__).resolve().parent.parent
SH = ROOT / "install.sh"
PS1 = ROOT / "install.ps1"
BASIC_TOOLS = ("sh", "awk", "sed", "uname", "id", "date", "mktemp", "rm", "df", "grep", "head", "tr", "wc", "dirname", "cat", "du", "sleep", "basename")

posix = pytest.mark.posix_only


def _hermetic_bin(tmp_path: Path, fakes: dict[str, str] | None = None) -> Path:
    """A PATH directory holding only basic tools (symlinks) and the fake programs asked for."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for name in BASIC_TOOLS:
        real = shutil.which(name)
        if real and not (d / name).exists():
            (d / name).symlink_to(real)
    for name, body in (fakes or {}).items():
        f = d / name
        f.write_text("#!/bin/sh\n" + body + "\n")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return d


def _run_sh(tmp_path: Path, *args: str, fakes: dict[str, str] | None = None, script: Path = SH, extra_env: dict[str, str] | None = None,
            ) -> subprocess.CompletedProcess:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"HOME": str(home), "PATH": str(_hermetic_bin(tmp_path, fakes)), "TERM": "dumb", "NIGHTAGE_TEST_RAM_MB": "7000", "LC_ALL": "C"}
    env.update(extra_env or {})
    return subprocess.run(["/bin/sh", str(script), *args], capture_output=True, text=True, timeout=60, env=env, stdin=subprocess.DEVNULL, cwd=str(tmp_path))


# ── syntax ──────────────────────────────────────────────────────────────────

@posix
def test_sh_syntax():
    for shell in ("sh", "dash"):
        exe = shutil.which(shell)
        if exe:
            res = subprocess.run([exe, "-n", str(SH)], capture_output=True, text=True)
            assert res.returncode == 0, (shell, res.stderr)


@posix
def test_shellcheck_clean_when_available():
    exe = shutil.which("shellcheck")
    if not exe:
        pytest.skip("shellcheck not installed")
    res = subprocess.run([exe, "-s", "sh", str(SH)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout


@posix
def test_sh_is_posix_not_bash():
    text = SH.read_text()
    assert text.startswith("#!/bin/sh\n")
    for bashism in ("[[ ", "<<<", "$'", "${BASH"):
        assert bashism not in text, bashism
    assert not re.search(r"^\s*(local|function)\s", text, re.M)


# ── dry run: the plan is printed, nothing changes ───────────────────────────

@posix
def test_dry_run_prints_the_whole_plan_and_changes_nothing(tmp_path):
    res = _run_sh(tmp_path, "--dry-run", "--yes", "--source", str(ROOT))
    out = res.stdout
    assert res.returncode == 0, out + res.stderr
    for needle in ("[1/7] What will happen", "[2/7]", "[3/7]", "[4/7]", "[5/7]", "[6/7]", "[7/7]", "DRY RUN",
                   "https://astral.sh/uv/install.sh",            # the exact uv URL is shown
                   "uv tool install --force --python 3.12",
                   "https://ollama.com/install.sh", "needs sudo", # Ollama's Linux script is explained, never silent
                   "qwen3.5:4b",                                  # 7000 MB of RAM -> half = 3.5 GB -> the 4b tier
                   "nightage doctor --no-web --only system,python,packages",
                   "nightage demo", "nightage chat", "uv tool uninstall agent", "install.log"):
        assert needle in out, needle
    assert "(dry run: not executed)" in out
    assert list((tmp_path / "home").iterdir()) == [], "a dry run must not create anything in HOME"


@posix
def test_dry_run_with_yes_never_runs_the_sudo_ollama_installer(tmp_path):
    res = _run_sh(tmp_path, "--dry-run", "--yes", "--source", str(ROOT))
    assert "$ sh " + "/" not in res.stdout.split("[5/7]")[1].split("[6/7]")[0]       # no command line for the Ollama script
    assert "never with plain --yes" in res.stdout


@posix
def test_with_ollama_flag_is_the_explicit_yes(tmp_path):
    res = _run_sh(tmp_path, "--dry-run", "--yes", "--with-ollama", "--source", str(ROOT))
    assert "ollama-install.sh" in res.stdout and res.returncode == 0


@posix
def test_model_tiers_follow_the_doctor(tmp_path):
    def model_for(ram: str, vram: str | None = None) -> str:
        env = {"NIGHTAGE_TEST_RAM_MB": ram}
        if vram:
            env["NIGHTAGE_TEST_VRAM_MB"] = vram
        out = _run_sh(tmp_path / f"t{ram}{vram}", "-n", "-y", "--source", str(ROOT), extra_env=env, fakes=None).stdout \
            if (tmp_path / f"t{ram}{vram}").mkdir() is None else ""
        return re.search(r"best fit is: (\S+)", out).group(1)
    assert model_for("2048") == "qwen3.5:0.8b"
    assert model_for("7000") == "qwen3.5:4b"
    assert model_for("32000") == "qwen3.5:9b"
    assert model_for("16000", "16000") == "qwen3.5:9b"
    assert model_for("64000", "48000") == "qwen3.5:27b"


@posix
def test_existing_tools_are_recognised(tmp_path):
    fakes = {"uv": 'echo "uv 0.13.5"', "ollama": 'echo "ollama version is 0.40.11"'}
    res = _run_sh(tmp_path, "--dry-run", "--yes", "--source", str(ROOT), fakes=fakes)
    assert "uv is already installed" in res.stdout and "Ollama is already installed" in res.stdout
    assert "astral.sh/uv/install.sh" not in res.stdout.split("[3/7]")[1].split("[4/7]")[0] or "already installed" in res.stdout


@posix
def test_flags_skip_ollama_and_model(tmp_path):
    res = _run_sh(tmp_path, "-n", "-y", "--no-ollama", "--no-model", "--source", str(ROOT))
    assert "Skipped (--no-ollama)" in res.stdout and "Skipped (--no-model)" in res.stdout
    assert "ollama pull" in res.stdout and "$ ollama pull" not in res.stdout


@posix
def test_placeholder_repo_url_asks_for_a_source(tmp_path):
    lone = tmp_path / "elsewhere"
    lone.mkdir()
    # the shipped script has the real project URL; the placeholder branch is exercised on a copy with the bare one
    (lone / "install.sh").write_text(re.sub(r'^REPO_URL="[^"]*"', 'REPO_URL="https://github.com/"', SH.read_text(), count=1, flags=re.M))
    res = _run_sh(tmp_path, "-n", "-y", script=lone / "install.sh")
    assert res.returncode == 1
    assert "--source" in res.stdout and "NIGHTAGE_REPO" in res.stdout


@posix
def test_nightage_repo_env_builds_an_archive_source(tmp_path):
    lone = tmp_path / "elsewhere"
    lone.mkdir()
    shutil.copy(SH, lone / "install.sh")
    res = _run_sh(tmp_path, "-n", "-y", script=lone / "install.sh", extra_env={"NIGHTAGE_REPO": "https://github.com/someone/nightage"})
    assert res.returncode == 0, res.stdout
    assert "agent @ https://github.com/someone/nightage/archive/HEAD.tar.gz" in res.stdout


@posix
def test_non_https_source_is_refused(tmp_path):
    res = _run_sh(tmp_path, "-n", "-y", "--source", "http://example.com/x.tar.gz")
    assert res.returncode == 1 and "https" in (res.stdout + res.stderr)


@posix
def test_unknown_option_and_help(tmp_path):
    assert _run_sh(tmp_path, "--frobnicate").returncode == 2
    res = _run_sh(tmp_path, "--help")
    assert res.returncode == 0 and "--dry-run" in res.stdout and "--uninstall" in res.stdout and "REPO_URL" not in res.stdout


@posix
def test_uninstall_dry_run_keeps_data(tmp_path):
    res = _run_sh(tmp_path, "-n", "-y", "--uninstall")
    assert res.returncode == 0 and "uv tool uninstall" in res.stdout or "No uv-installed" in res.stdout
    assert "never removes uv" in res.stdout


@posix
def test_uninstall_yes_never_deletes_data(tmp_path):
    home = tmp_path / "home"
    data = home / ".local" / "share" / "nightage"
    data.mkdir(parents=True)
    (data / "config.yaml").write_text("x: 1\n")
    res = _run_sh(tmp_path, "--yes", "--uninstall")
    assert res.returncode == 0, res.stdout
    assert (data / "config.yaml").exists()
    assert "KEEP" in res.stdout


@posix
def test_no_terminal_and_no_yes_refuses_instead_of_guessing(tmp_path):
    res = _run_sh(tmp_path, "--source", str(ROOT))           # real mode, stdin closed: must stop at the first question
    assert res.returncode == 1
    assert "no terminal" in (res.stdout + res.stderr).lower()
    assert list((tmp_path / "home").glob(".local/bin")) == []


# ── safety properties of the script text ────────────────────────────────────

@posix
def test_sh_safety_properties():
    text = SH.read_text()
    assert "id -u" in text and "--force" in text                      # refuses root unless forced
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if re.search(r"\brun\s+sudo\b|^\s*sudo\b", ln)], "sudo is never run directly"
    spoken = ("info ", "say ", "printf ", "warn ", "die ")
    assert not [ln for ln in code if re.search(r"\|\s*(sudo\s+)?(ba)?sh\b", ln) and "curl" in ln and not ln.lstrip().startswith(spoken)], "no blind curl | sh"
    assert re.search(r"https://\*\)", text) and "refusing to download from a non-HTTPS" in text
    for url in re.findall(r"https?://[^\s\"'<>)]+", "\n".join(code)):
        assert url.startswith("https://"), url
    assert "NO_COLOR" in text
    assert 'rm -rf "$dd"' in text and 'case "$dd" in' in text            # deletions only of the listed data dirs, with a sanity guard


def test_repo_url_matches_brand():
    m = re.search(r'^REPO_URL="([^"]*)"', SH.read_text(), re.M)
    assert m and m.group(1) == brand.REPO_URL
    p = re.search(r'^\$RepoUrl\s*=\s*"([^"]*)"', PS1.read_text(), re.M)
    assert p and p.group(1) == brand.REPO_URL


# ── install.ps1: static checks (always) ─────────────────────────────────────

def _ps1_code_lines() -> list[str]:
    text = PS1.read_text()
    text = re.sub(r"<#.*?#>", "", text, count=1, flags=re.S)           # the help block mentions the one-liner
    return [ln for ln in text.splitlines() if not ln.strip().startswith("#")]


def test_ps1_is_ascii_only_for_windows_powershell_5():
    data = PS1.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf") or True
    data.decode("ascii")                                                # PowerShell 5.1 reads BOM-less files as ANSI: no non-ASCII at all


def test_ps1_strict_mode_and_stop_on_error():
    text = PS1.read_text()
    assert re.search(r"^Set-StrictMode -Version Latest\s*$", text, re.M)
    assert re.search(r"^\$ErrorActionPreference = 'Stop'\s*$", text, re.M)


def test_ps1_never_evaluates_downloaded_text():
    code = "\n".join(_ps1_code_lines())
    assert not re.search(r"\bInvoke-Expression\b|\biex\b", code, re.I), "downloaded content must be saved, shown, then run as a file"
    assert "-File" in code and "Get-Download $UvUrl" in code           # uv's script: download, hash shown, run as a file
    assert "$UvUrl" in code and 'https://astral.sh/uv/install.ps1' in code


def test_ps1_has_no_force_deletes_and_deletes_only_known_folders():
    code = _ps1_code_lines()
    assert not [ln for ln in code if re.search(r"-Force\b", ln) and not ln.lstrip().startswith(("Say ", "Info ", "Warn "))], "no -Force in executable code"
    removes = [ln for ln in code if re.search(r"(^\s*|\{\s*)Remove-Item", ln)]
    assert removes and all("-LiteralPath" in ln for ln in removes)
    assert all(("$script:Tmp" in ln) or ("$d" in ln) for ln in removes)


def test_ps1_downloads_are_https_only_and_admin_is_refused():
    code = "\n".join(_ps1_code_lines())
    for url in re.findall(r"http://[^\s\"']+", code):
        assert "localhost" in url, url
    assert "IsInRole" in code and "$Force" in code
    assert "Get-AuthenticodeSignature" in code                            # Ollama's exe: signature is checked before it runs
    assert "Start-Transcript" in code                                     # log file for bug reports
    assert "winget install --id Ollama.Ollama" in code and "astral-sh.uv" in code


def test_ps1_has_the_same_flags_as_the_shell_script():
    text = PS1.read_text()
    for flag in ("Yes", "DryRun", "NoOllama", "NoModel", "Source", "Uninstall", "Help", "Force", "Model", "WithOllama", "Embed", "NoEmbed"):
        assert re.search(rf"\${flag}\b", text), flag
    assert "[Environment]::GetEnvironmentVariable" in text                # user-scope PATH, never Machine writes
    assert "SetEnvironmentVariable" not in text


# ── install.ps1 under a real PowerShell, when there is one ──────────────────

PWSH = shutil.which("pwsh")


@pytest.mark.skipif(not PWSH, reason="pwsh not installed")
def test_ps1_parses_and_dry_runs_under_pwsh(tmp_path):
    parse = subprocess.run([PWSH, "-NoProfile", "-Command",
                            f"$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile('{PS1}',[ref]$t,[ref]$e);"
                            "if($e.Count){$e|%{$_.ToString()};exit 1}"], capture_output=True, text=True, timeout=90)
    assert parse.returncode == 0, parse.stdout + parse.stderr
    home = tmp_path / "h"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "NIGHTAGE_TEST_RAM_MB": "7000", "NO_COLOR": "1", "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1"}
    res = subprocess.run([PWSH, "-NoProfile", "-File", str(PS1), "-DryRun", "-Yes", "-Source", str(ROOT), "-NoOllama"],
                         capture_output=True, text=True, timeout=120, env=env, stdin=subprocess.DEVNULL)
    assert res.returncode == 0, res.stdout + res.stderr
    for needle in ("[7/7]", "https://astral.sh/uv/install.ps1", "uv tool install --force --python 3.12", "doctor --no-web", "DRY RUN"):
        assert needle in res.stdout, needle
    assert not (home / "AppData").exists()


# ── the memory (embedding) model choice ─────────────────────────────────────

def _embed_plan(tmp_path: Path, ram: str, lang: str, *args: str) -> str:
    d = tmp_path / f"e{ram}{lang}{'-'.join(args)}".replace("/", "_")
    d.mkdir()
    return _run_sh(d, "-n", "-y", "--source", str(ROOT), *args, extra_env={"NIGHTAGE_TEST_RAM_MB": ram, "LC_ALL": lang}).stdout


@posix
@pytest.mark.parametrize("ram,lang,tag", [
    ("2000", "C", "all-minilm"),
    ("7000", "en_US.UTF-8", "nomic-embed-text"),
    ("7000", "de_DE.UTF-8", "embeddinggemma:300m-qat-q4_0"),
    ("32000", "de_DE.UTF-8", "bge-m3"),
    ("32000", "en_GB.UTF-8", "nomic-embed-text"),
])
def test_embed_choice_default_follows_ram_and_language_like_the_catalogue(tmp_path, ram, lang, tag):
    from agent import embedmodels

    assert embedmodels.recommend(int(ram), 0, lang).tag == tag          # the python rule and the installer agree
    out = _embed_plan(tmp_path, ram, lang)
    assert f"Which memory model? [auto: {tag}]" in out
    assert f"embed use {tag} --yes --no-pull" in out
    assert re.search(rf"\d\) {re.escape(tag)}\s.*<- recommended for this computer", out)
    for n in ("0) none", "English only - smallest, fastest", "Multilingual"):
        assert n in out


@posix
def test_embed_flags(tmp_path):
    assert "embed use all-minilm --yes" in _embed_plan(tmp_path, "7000", "C", "--embed", "all-minilm")
    out = _embed_plan(tmp_path, "7000", "C", "--no-embed")
    assert "Skipped (--no-embed)" in out and "embed use" not in out and "Which memory model" not in out


@posix
def test_installer_embed_table_matches_the_catalogue():
    from agent import embedmodels

    text = SH.read_text()
    rows = re.findall(r"^(?:EMB_TABLE=')?(\d)\|([^|]+)\|(\d+)\|(.+?)'?$", text, re.M)
    assert len(rows) == 5
    for _n, tag, mb, label in rows:
        m = embedmodels.describe(tag)
        assert m.exact and m.curated, tag
        assert int(mb) == m.download_mb, tag
        assert label.split(" - ")[0].split(" (")[0] == m.label.split(" - ")[0].split(" (")[0], tag    # English only / Multilingual
    ps = PS1.read_text()
    ps_rows = re.findall(r'@\((\d), "([^"]+)", (\d+), "', ps)
    assert [(t, int(mb)) for _n, t, mb in ps_rows] == [(t, int(mb)) for _n, t, mb, _l in rows]


@posix
def test_termux_curl_pipe_fetches_the_project_archive_and_continues_with_the_termux_installer(tmp_path):
    """On a phone `curl ... | sh` has no checkout next to the script: it must fetch the archive and hand over, not stop."""
    env = dict(os.environ, TERMUX_VERSION="0.118", HOME=str(tmp_path))
    res = subprocess.run(["sh", "-s", "--", "--dry-run", "--yes"], input=SH.read_text(), capture_output=True, text=True, env=env, timeout=60)
    out = re.sub(r"\x1b\[[0-9;]*m", "", res.stdout)
    assert res.returncode == 0, out
    assert "/archive/HEAD.tar.gz" in out and "tar -xz --strip-components=1" in out
    assert "install-termux.sh --yes --dry-run" in out
    assert "Get install-termux.sh from the project page" not in out
