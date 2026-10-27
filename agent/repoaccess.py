"""Per-folder permission levels for autonomous work, set with slash commands.

    work in this repo: /home/me/project   -> the agent asks which level to use
    /auto-/home/me/project      full: edits AND shell commands inside the folder run without asking
    /medium-/home/me/project    medium: edits inside the folder run without asking, commands still ask
    /ask-/home/me/project       ask for everything again (revokes a grant for this session)
    /greenlist-/home/me/project like /auto, but remembered forever (~/.local/state/agent/greenlist.json)
    /ungreenlist-/home/me/project, /greenlist (show the list), /access (show what is granted now)

Grants only ever *skip the y/yes prompt* of calls the risk classifier rated CAUTION. BLOCKED and
DANGEROUS calls (and every tool other than write_file / replace_in_file / run_command) still behave
exactly as before. Shell commands in /auto mode are additionally scoped: the directory they run in must
be inside the folder and every absolute or ~ path in the command must be inside it too (or a harmless
system location). That is a guard rail against a model slip, not a sandbox: a script run inside the
folder can still do anything a script can.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

from agent import osport
from typing import Any

ASK, MEDIUM, AUTO = "ask", "medium", "auto"
_RANK = {ASK: 0, MEDIUM: 1, AUTO: 2}
EDIT_TOOLS = ("write_file", "replace_in_file")
SHELL_TOOL = "run_command"

_FORBIDDEN_TOP = frozenset({"etc", "usr", "bin", "sbin", "lib", "lib64", "boot", "var", "nix", "sys", "proc",
                            "dev", "root", "run"})
_HARMLESS_PREFIXES = ("/usr/", "/bin/", "/sbin/", "/lib/", "/nix/store/", "/run/current-system/", "/tmp/",
                      "/dev/null", "/dev/stdout", "/dev/stderr", "/proc/cpuinfo", "/proc/meminfo",
                      # Termux's read-only system prefix (its /usr): additive, same class of path as /usr/ above
                      "/data/data/com.termux/files/usr/")

_session: dict[Path, str] = {}

COMMAND_RE = re.compile(r"^/(auto|medium|ask|greenlist|ungreenlist)(?:[-\s]\s*(.+))?$", re.IGNORECASE)
WORK_IN_RE = re.compile(
    r"\b(?:work in|work on|arbeite (?:in|im|an))\s+(?:this\s+|diesem\s+|dem\s+)?"
    r"(?:repo|repository|folder|directory|ordner|verzeichnis)?\s*:?\s*[\"']?(~?/[^\s\"']+)",
    re.IGNORECASE,
)


# ------------------------------------------------------------------------------------ persistence


def _greenlist_file() -> Path:
    base = str(osport.state_home())
    return Path(base) / "agent" / "greenlist.json"


def greenlist() -> list[Path]:
    try:
        data = json.loads(_greenlist_file().read_text())
        return [Path(p) for p in data.get("paths", []) if isinstance(p, str)]
    except (OSError, ValueError, AttributeError):
        return []


def _save_greenlist(paths: list[Path]) -> None:
    f = _greenlist_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"paths": sorted({str(p) for p in paths})}, indent=2) + "\n")


def reset_session() -> None:
    _session.clear()


# ----------------------------------------------------------------------------------------- folders


def _case_insensitive(path: Path) -> Path | None:
    """Fix wrong capitalisation (`/home/me/downloads` -> `/home/me/Downloads`) if it is unambiguous."""
    cur = Path(path.anchor or "/")
    for part in path.parts[1:]:
        nxt = cur / part
        if nxt.exists():
            cur = nxt
            continue
        try:
            matches = [c for c in cur.iterdir() if c.name.lower() == part.lower()]
        except OSError:
            return None
        if len(matches) != 1:
            return None
        cur = matches[0]
    return cur


def folder_problem(folder: Path) -> str | None:
    """Why a folder must never be granted, or None. The home directory and top-level folders are out."""
    if folder == Path.home() or len(folder.parts) < 4:
        return "that is your home directory or a top-level folder - pick a project folder inside it"
    if folder.parts[1] in _FORBIDDEN_TOP and not str(folder).startswith(str(Path.home())):
        return "system folders can't be granted"
    return None


def resolve_folder(text: str) -> tuple[Path | None, str]:
    """(folder, message). `folder` is None with an explanation when the text can't be used."""
    raw = (text or "").strip().strip("\"'")
    if not raw:
        return None, "give a folder, e.g. /auto-/home/you/project"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        return None, f"{raw!r} is not an absolute path"
    note = ""
    if not p.exists():
        fixed = _case_insensitive(p)
        if fixed is None:
            return None, f"{raw} does not exist"
        note = f" (using {fixed})"
        p = fixed
    try:
        p = p.resolve()
    except OSError:
        return None, f"can't resolve {raw}"
    if not p.is_dir():
        return None, f"{p} is not a folder"
    problem = folder_problem(p)
    if problem:
        return None, f"{p}: {problem}"
    return p, note


# ------------------------------------------------------------------------------------------ grants


def grant(folder: Path, mode: str) -> None:
    if mode == ASK:
        _session.pop(folder, None)
    else:
        _session[folder] = mode


def mode_for(path: Path) -> str:
    """Highest mode of any grant (session or greenlist) whose folder contains `path`."""
    best = ASK
    for folder, mode in [*_session.items(), *((g, AUTO) for g in greenlist())]:
        if (path == folder or folder in path.parents) and _RANK[mode] > _RANK[best]:
            best = mode
    return best


def _under(path: Path, folder: Path) -> bool:
    return path == folder or folder in path.parents


_CD_UNKNOWN = re.compile(r"[$`()<>|&;*?\[\]{}\\\n]")


def _shell_in_scope(command: str, folder: Path, cwd: Path | None = None) -> bool:
    """True if a shell command only touches `folder` (plus harmless system locations). `cwd` is where it runs.
    A plain `cd <path>` / `cd ..` step of an `&&` / `;` chain is followed: later relative paths (`../main.py`) are judged from where the
    chain has got to. A `cd` that cannot be followed (variable, substitution, option, several operands) makes the answer False."""
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)    # `a;b` and `a&&b` split without spaces, paths stay whole
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return False
    here = (cwd or Path.cwd())
    try:
        here = here.resolve()
    except OSError:
        return False
    chained = True                                   # every separator so far was `&&`: a failed `cd` ended the chain
    segment: list[str] = []
    for tok in [*tokens, ";"]:
        if tok not in ("&&", ";"):
            segment.append(tok)
            continue
        if segment and segment[0] == "cd":
            if len(segment) != 2 or segment[1].startswith("-") or _CD_UNKNOWN.search(segment[1]):
                return False
            target = Path(os.path.expanduser(segment[1]))
            try:
                target = (target if target.is_absolute() else here / target).resolve()
            except (OSError, RuntimeError):
                return False
            if not _under(target, folder) or not (target.is_dir() or chained):
                return False                         # outside, or a missing folder after `;` (the chain would go on elsewhere)
            here = target
        elif segment and not (_under(here, folder) and _paths_in_scope(segment, folder, here)):       # a leading `cd /elsewhere` is judged on its own
            return False
        segment = []
        chained = chained and tok == "&&"
    return True


def _paths_in_scope(tokens: list[str], folder: Path, here: Path) -> bool:
    for tok in tokens:
        for piece in re.split(r"[=:,]", tok):
            if ".." in Path(piece).parts:
                try:
                    if not _under((here / os.path.expanduser(piece)).resolve(), folder):
                        return False
                except (OSError, RuntimeError):
                    return False
            if piece.startswith(("/", "~")):
                target = Path(os.path.expanduser(piece))
                try:
                    target = target.resolve()
                except OSError:
                    return False
                if _under(target, folder) or str(target).startswith(_HARMLESS_PREFIXES) or piece.startswith(_HARMLESS_PREFIXES):
                    continue
                return False
    return True


def allows(name: str, arguments: dict[str, Any], level_is_caution: bool) -> bool:
    """Does a folder grant let this CAUTION-rated call run without the y/yes prompt?"""
    if not level_is_caution:
        return False
    if name in EDIT_TOOLS and arguments.get("path"):
        try:
            target = Path(str(arguments["path"])).expanduser().resolve()
        except (OSError, ValueError):
            return False
        return _RANK[mode_for(target)] >= _RANK[MEDIUM]
    if name == SHELL_TOOL:
        command = str(arguments.get("command", ""))
        folders = [f for f, m in [*_session.items(), *((g, AUTO) for g in greenlist())] if m == AUTO]
        return any(_shell_in_scope(command, f) for f in folders)
    return False


# --------------------------------------------------------------------------------------- commands


def parse_work_in(text: str) -> str | None:
    m = WORK_IN_RE.search(text or "")
    return m.group(1) if m else None


def question(folder: Path, note: str = "") -> str:
    return (
        f"Working in {folder}{note}. How much may I do there without asking?\n"
        f"  /auto-{folder}      full - edit files and run commands in this folder\n"
        f"  /medium-{folder}    medium - edit files freely, ask before running commands\n"
        f"  /ask-{folder}       ask before everything (default)\n"
        f"  /greenlist-{folder} like auto, and remembered for every future session"
    )


def handle_command(user_input: str) -> tuple[str, str] | None:
    """Run an access slash command. Returns (kind, message) where kind is the command word, or None
    if the input is not an access command. `kind` is 'auto'/'medium'/'ask' when a level was applied."""
    text = user_input.strip()
    if text.lower() in ("/access", "/greenlist", "/greenlist-"):
        lines = ["Session grants:" if _session else "Session grants: none"]
        lines += [f"  {m:<7} {f}" for f, m in sorted(_session.items(), key=lambda kv: str(kv[0]))]
        gl = greenlist()
        lines.append("Greenlist (always full access):" if gl else "Greenlist: empty")
        lines += [f"  {p}" for p in gl]
        return "access", "\n".join(lines)
    m = COMMAND_RE.match(text)
    if not m:
        return None
    word, arg = m.group(1).lower(), m.group(2)
    folder, note = resolve_folder(arg or "")
    if folder is None:
        return word, f"{word}: {note}"
    if word in (AUTO, MEDIUM, ASK):
        grant(folder, word)
        text_by_mode = {
            AUTO: "full access: I edit files and run commands in",
            MEDIUM: "medium access: I edit files freely in (commands still ask)",
            ASK: "back to asking for everything in",
        }
        return word, f"{text_by_mode[word]} {folder}{note}"
    if word == "greenlist":
        paths = greenlist()
        if folder not in paths:
            paths.append(folder)
            _save_greenlist(paths)
        return word, f"greenlisted {folder}{note} - full access in every future session (remove with /ungreenlist-{folder})"
    paths = [p for p in greenlist() if p != folder]
    _save_greenlist(paths)
    return word, f"removed {folder} from the greenlist"
