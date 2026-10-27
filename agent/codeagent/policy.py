"""Per-repo freedom for the coding agent: how much it may do in a folder, without asking.

A repo is configured in `~/.config/agent/code-policy.json` (or with the `/green` and `/allow` commands inside the
coding session). Only YOUR global file grants anything: a policy file inside a repo is never read, so a cloned
repo cannot hand itself permissions.

    mode        ask      every edit and command asks (default)
                medium   edits inside the repo run freely, commands ask unless they match `allow`
                auto     edits and commands inside the repo run freely (the "green repo")
    allow       command patterns that never ask (glob, e.g. "pytest*", "npm run *", "cargo test*")
    deny        command patterns that never run (glob), on top of the built-in safety classifier
    network     may commands reach the network (curl, wget, git fetch/pull/push/clone)?
    installs    may commands install packages (pip/npm/cargo/apt)? Needs network too for remote installs
    commit      never | ask | auto   - may the agent `git commit` its work?
    max_steps   tool steps per prompt before it asks whether to continue
    think       auto | on | off      - reasoning mode for tool steps (auto = off, on only when stuck/planning)
    model, ctx, test_cmd   per-repo overrides

Anything the safety classifier rates BLOCKED never runs, in every mode. Everything the agent does still happens inside
the repo folder: paths outside it (and commands that reach outside it) ask, whatever the mode.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from agent import osport
from typing import Any, Callable, Mapping

ASK, MEDIUM, AUTO = "ask", "medium", "auto"
MODES = (ASK, MEDIUM, AUTO)
_RANK = {ASK: 0, MEDIUM: 1, AUTO: 2}

NETWORK_RE = re.compile(r"\b(curl|wget|nc|ncat|ssh|scp|rsync|ftp|git\s+(?:fetch|pull|push|clone|remote\s+add))\b")
INSTALL_RE = re.compile(r"\b(pip3?\s+install|python3?\s+-m\s+pip\s+install|uv\s+(?:pip\s+install|add|sync)|npm\s+(?:i|install|ci)|"
                        r"pnpm\s+(?:i|install|add)|yarn\s+(?:add|install)|cargo\s+(?:add|install)|go\s+(?:get|install)|"
                        r"apt(?:-get)?\s+install|nix-env\s+-i|nix\s+profile\s+install)\b")
PUSH_RE = re.compile(r"\bgit\s+push\b|\bgh\s+(?:pr|release|repo)\s+(?:create|merge|delete)\b")


def policy_file() -> Path:
    return Path(os.environ.get("AGENT_CODE_POLICY") or Path.home() / ".config" / "agent" / "code-policy.json") if sys.platform != "win32" else Path(os.environ.get("AGENT_CODE_POLICY") or osport.config_home() / "agent" / "code-policy.json")


@dataclass
class RepoPolicy:
    mode: str = ASK
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    network: bool = False
    installs: bool = False
    commit: str = "ask"
    max_steps: int = 60                 # the user's cap when steps_set, else only the ceiling of the automatic step budget (stepbudget.py)
    steps_set: bool = False             # True once max_steps was given explicitly (per-folder policy, CLI, web form)
    think: str = "auto"
    model: str | None = None
    ctx: int | None = None
    test_cmd: str | None = None

    def clean(self) -> "RepoPolicy":
        if self.mode not in MODES:
            self.mode = ASK
        if self.commit not in ("never", "ask", "auto"):
            self.commit = "ask"
        if self.think not in ("auto", "on", "off"):
            self.think = "auto"
        self.max_steps = max(5, min(int(self.max_steps or 60), 500))
        self.allow = [str(x) for x in self.allow if str(x).strip()]
        self.deny = [str(x) for x in self.deny if str(x).strip()]
        return self


@dataclass
class Decision:
    verdict: str                  # allow | ask | block
    reason: str = ""


def _load_raw() -> dict[str, Any]:
    try:
        data = json.loads(policy_file().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _from_dict(d: dict[str, Any], base: RepoPolicy | None = None) -> RepoPolicy:
    pol = RepoPolicy(**asdict(base)) if base else RepoPolicy()
    known = {f.name for f in fields(RepoPolicy)}
    for k, v in (d or {}).items():
        if k in known:
            setattr(pol, k, v)
    if "max_steps" in (d or {}):
        pol.steps_set = True
    return pol.clean()


def load_defaults() -> RepoPolicy:
    return _from_dict(_load_raw().get("defaults") or {})


def for_repo(root: Path | str) -> RepoPolicy:
    """The effective policy of `root`: defaults, then the repo's entry. The chat greenlist (/greenlist) counts as auto."""
    root = Path(root).resolve()
    raw = _load_raw()
    pol = _from_dict(raw.get("defaults") or {})
    best, best_len = None, -1
    for key, entry in (raw.get("repos") or {}).items():           # the most specific enclosing folder wins
        k = Path(key).expanduser()
        if (root == k or k in root.parents) and len(str(k)) > best_len and isinstance(entry, dict):
            best, best_len = entry, len(str(k))
    if best:
        pol = _from_dict(best, pol)
    try:
        from agent import repoaccess

        if _RANK[repoaccess.mode_for(root)] > _RANK[pol.mode]:
            pol.mode = repoaccess.mode_for(root)
    except Exception:  # noqa: BLE001 - the chat greenlist is a bonus
        pass
    return pol


def save_repo(root: Path | str, **changes: Any) -> RepoPolicy:
    """Persist changes for one repo (merged into its entry) and return the effective policy."""
    root = Path(root).resolve()
    raw = _load_raw()
    repos = raw.setdefault("repos", {})
    entry = dict(repos.get(str(root)) or {})
    known = {f.name for f in fields(RepoPolicy)}
    for k, v in changes.items():
        if k in known:
            entry[k] = v
    cleaned = asdict(_from_dict(entry, RepoPolicy()))
    repos[str(root)] = {k: cleaned[k] for k in entry if k in cleaned}      # sparse: unset keys keep inheriting the defaults
    f = policy_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    return for_repo(root)


def forget_repo(root: Path | str) -> bool:
    root = str(Path(root).resolve())
    raw = _load_raw()
    if root in (raw.get("repos") or {}):
        del raw["repos"][root]
        policy_file().write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        return True
    return False


def all_repos() -> dict[str, RepoPolicy]:
    return {k: _from_dict(v) for k, v in (_load_raw().get("repos") or {}).items() if isinstance(v, dict)}


# ── decisions ───────────────────────────────────────────────────────────────

def _matches(command: str, patterns: list[str]) -> str | None:
    cmd = command.strip()
    for pat in patterns:
        if fnmatch.fnmatch(cmd, pat) or fnmatch.fnmatch(cmd, pat + " *") or cmd == pat:
            return pat
    return None


def inside(path: Path, root: Path) -> bool:
    try:
        p = path.resolve()
    except OSError:
        return False
    return p == root or root in p.parents


TEMP_DIRS = ("/tmp", "/var/tmp", "/dev/shm")
TEMP_RE = re.compile(r"(?<![\w.\-/])(?:/var/tmp|/tmp|/dev/shm)(?![\w.\-])|\$\{?TMPDIR\b|\bmktemp\b")
TEMP_REASON = "temp folders are off limits - use .scratch/ inside the repo"


# ── the jail: private / system folders are a hard NO ────────────────────────
# Not "ask": a prompt for ~/.ssh stalls an unattended run and teaches nothing. Built-in list below; `coding.deny_paths` can only
# ADD to it and `coding.allow_paths` only adds READ access to paths that were merely "outside the repo" (it never lifts a deny).

_HOME_DENY = (".ssh", ".gnupg", ".aws", ".kube", ".docker", ".password-store", ".config", ".local/share", ".claude", ".mozilla",
              ".thunderbird", ".librewolf", ".var", "snap", ".netrc", ".git-credentials", ".npmrc", ".pypirc", ".bash_history",
              ".zsh_history", ".python_history", ".bashrc", ".bash_profile", ".bash_logout", ".profile", ".zshrc", ".zprofile", ".zshenv")
_HOME_EXCEPT = (".local/share/uv",)
_SYS_DENY = ("/etc/shadow", "/etc/gshadow", "/etc/sudoers", "/etc/ssh", "/etc/ssl/private", "/root", "/boot", "/sys", "/proc", "/dev")
_SYS_PREFIX = ("/etc/shadow", "/etc/gshadow", "/etc/sudoers")          # also shadow-, sudoers.d ...
_LITERAL_OK = frozenset({"/dev/null", "/dev/zero", "/dev/urandom", "/dev/random", "/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/tty",
                         "/proc/cpuinfo", "/proc/meminfo"})
_EXTRA: dict[str, list[str]] = {"deny": [], "allow": []}


def _home() -> str:
    return os.path.realpath(str(Path.home()))


def _under(p: str, base: str) -> bool:
    return p == base or p.startswith(base.rstrip("/") + "/")


def _clean_paths(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for v in value:
        s = _expand(v) if isinstance(v, str) else None
        if s and os.path.isabs(s):
            out.append(os.path.normpath(s))
    return out


def configure(config: dict[str, Any] | None) -> None:
    """Read `coding.deny_paths` / `coding.allow_paths` (extend-only; a garbage value is ignored). Called once per engine."""
    c = (config or {}).get("coding") or {}
    c = c if isinstance(c, dict) else {}
    _EXTRA["deny"], _EXTRA["allow"] = _clean_paths(c.get("deny_paths")), _clean_paths(c.get("allow_paths"))


def _expand(raw: str) -> str | None:
    """`~`, `$HOME`, `${HOME}` expanded; None when another variable / substitution makes the path unknowable."""
    s = str(raw).strip().strip("\"'")
    if not s:
        return None
    home = _home()
    s = re.sub(r"\$\{HOME\}|\$HOME\b", home, s)
    if s == "~" or s.startswith("~/"):
        s = home + s[1:]
    elif s.startswith("~"):
        s = os.path.expanduser(s)
    return None if ("$" in s or "`" in s or "\0" in s) else s


def _private(p: str, home: str, rootp: str) -> bool:
    """Is the (normalised or fully resolved) absolute path `p` off limits? Paths inside the repo are the repo's own business."""
    if p in _LITERAL_OK:
        return False
    if _under(p, "/dev/shm"):
        return False                                                   # temp rule handles it, with its own message
    if _under(p, rootp) and not _under(home, rootp):
        return False                                                   # (a repo that contains the home dir gets no exemption)
    if any(_under(p, d) for d in _EXTRA["deny"]):
        return True
    if p == home:
        return rootp != home
    if _under(p, home):
        rel = os.path.relpath(p, home)
        if any(_under(rel, e) for e in _HOME_EXCEPT):
            return False
        return any(_under(rel, d) for d in _HOME_DENY)
    if any(_under(p, d) for d in _SYS_DENY) or p.startswith(_SYS_PREFIX):
        return True
    if os.path.dirname(home) == "/home":                               # other users' homes
        return p == "/home" or _under(p, "/home")
    return False


def forbidden_path(path: str, root: Path | str) -> str | None:
    """None, or the one-sentence NO for a path the agent must never touch. `~`, `$HOME`, `..` and symlinks are resolved first;
    relative paths count from `root`."""
    s = _expand(path)
    if s is None:
        return None
    rootp = os.path.realpath(str(root))
    joined = s if os.path.isabs(s) else os.path.join(rootp, s)
    home = _home()
    norm = os.path.normpath(joined)
    if norm in _LITERAL_OK:
        return None                                                    # /dev/stdout is a symlink into /proc: judge the name, not the target
    if _private(norm, home, rootp) or _private(os.path.realpath(joined), home, rootp):
        shown = str(path).strip()
        return f"NO: {shown[:80]} is private/system; work only inside the repo (scratch in .scratch/)."
    return None


def _readable_extra(target: Path) -> bool:
    p = os.path.realpath(str(target))
    return any(_under(p, a) for a in _EXTRA["allow"])


_ABS_RE = re.compile(r"(?<![\w.\-/~$])(?:~[\w.\-]*/[^\s;|&<>()`=,:]*|/[^\s;|&<>()`=,:]*)")
_PUNCT = frozenset(";&|")


def _command_paths(command: str, root: Path | str) -> list[str]:
    """Every path-like thing in a shell command, absolute and normalised: absolute / `~` / `$HOME` paths anywhere in the text
    (quotes dropped, so "$HOME"/.ssh and python -c strings count) plus each shell word resolved from the current folder, which a
    leading `cd` moves. Conservative on purpose: better a false NO than a read of ~/.ssh."""
    rootp = os.path.realpath(str(root))
    home = _home()
    text = re.sub(r"\$\{HOME\}|\$HOME\b", home, command).replace("\n", " ; ")
    found = [os.path.normpath(x) for x in (_expand(m.group(0)) for m in _ABS_RE.finditer(text.replace('"', "").replace("'", ""))) if x]
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=True)
        lex.whitespace_split, lex.commenters = True, ""
        toks = list(lex)
    except ValueError:
        toks = text.split()
    base, at_start, cd_next = rootp, True, False
    for t in toks:
        if t and set(t) <= _PUNCT | {"(", ")", "<", ">"}:
            at_start = t[0] in _PUNCT or t[0] == "("
            cd_next = False
            continue
        word = t.split("=", 1)[1] if t.startswith("-") and "=" in t else None if t.startswith("-") else t      # --file=/x -> /x; -n skipped
        if not word:
            at_start = False
            continue
        if at_start and t == "cd":
            cd_next, at_start = True, False
            continue
        at_start = False
        if any(ch.isspace() for ch in word):
            continue                                                   # python -c "..." and friends: the text scan covered it
        w = _expand(word)
        if w is None:
            continue
        p = os.path.normpath(w if os.path.isabs(w) else os.path.join(base, w))
        found.append(p)
        if cd_next:
            base, cd_next = os.path.realpath(p), False
    if toks and toks[-1] == "cd":
        found.append(home)                                             # bare `cd`
    return found


def forbidden_in_command(command: str, root: Path | str) -> str | None:
    """The first jailed path of a command; a simple variable assigned earlier in the same command line counts with its value
    (`X=~/.ssh; rm -rf $X`)."""
    resolved = resolve_vars(command) if "$" in command else None
    for text in (command, resolved) if resolved and resolved != command else (command,):
        for p in _command_paths(text, root):
            why = forbidden_path(p, root)
            if why:
                return why
    return None


def _scope_with_allow(command: str, root: Path | str) -> bool:
    """Auto mode: a command whose only outside paths are `coding.allow_paths` (or harmless system folders) stays in scope."""
    from agent import repoaccess

    if not _EXTRA["allow"]:
        return False
    rootp = os.path.realpath(str(root))
    for p in _command_paths(command, root):
        real = os.path.realpath(p)
        if _under(real, rootp) or any(_under(real, a) for a in _EXTRA["allow"]) or (real + "/").startswith(repoaccess._HARMLESS_PREFIXES) \
                or real.startswith(repoaccess._HARMLESS_PREFIXES):
            continue
        return False
    return True


def _root_in_temp(root: Path) -> bool:
    try:
        r = str(root.resolve())
    except OSError:
        return False
    return any(r == d or r.startswith(d + "/") for d in TEMP_DIRS)


def check_path(pol: RepoPolicy, root: Path, path: str, write: bool) -> Decision:
    target = (root / path).expanduser() if not os.path.isabs(os.path.expanduser(path)) else Path(os.path.expanduser(path))
    if not inside(target, root):
        if not _root_in_temp(root) and any(str(target).startswith(d + "/") or str(target) == d for d in TEMP_DIRS):
            return Decision("block", TEMP_REASON)
        why = forbidden_path(path, root)
        if why:
            return Decision("block", why)
        if not write and _readable_extra(target):
            return Decision("allow", "in coding.allow_paths")
        return Decision("ask", f"{path} is outside the repo")
    why = forbidden_path(path, root)               # only reached inside the repo: a repo that contains the home dir still has its secrets
    if why:
        return Decision("block", why)
    if ".git" in target.resolve().relative_to(root).parts[:1]:
        return Decision("ask" if not write else "block", "inside .git")
    if not write:
        return Decision("allow")
    return Decision("allow") if _RANK[pol.mode] >= _RANK[MEDIUM] else Decision("ask", "edits need approval in 'ask' mode")


DISPOSABLE_DIRS = {".scratch", "scratch", "tmp", ".tmp", "build", "dist", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "target"}
_VERB_FLAGS = {"rm": "rRfvdI", "unlink": "", "rmdir": "pv", "cp": "rRfapvnLPdHu", "mv": "fnvu", "mkdir": "pv", "touch": "c"}
_SEGMENT_SPLIT = re.compile(r"\s*(&&|;)\s*")
_UNSAFE = re.compile(r"[|`><$\\&\n\r!]|\bsudo\b|\.\.")       # substitutions, pipes, redirects, variables, escapes, background, parent dirs
_TMP = Path("/tmp")
_ASSIGN_TOK = re.compile(r"[A-Za-z_]\w*=")
_VAR_REF = re.compile(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)")
_BAD_VALUE = re.compile(r"[\s*?\[\]{}`$!\\;&|<>()]")           # a value that word-splits, globs or hides a substitution is not 'simple'
_SHELL_VARS = frozenset({"IFS", "PATH", "HOME", "PWD", "OLDPWD", "TMPDIR", "SHELL"})
_ROUGH_ASSIGN = re.compile(r"(?:^|[\s;&])[A-Za-z_]\w*=")


def _subst(text: str, env: dict[str, str]) -> str | None:
    """`$VAR` / `${VAR}` of `text` replaced from `env`; None for an unknown variable or any other `$` / backquote."""
    try:
        out = _VAR_REF.sub(lambda m: env[m.group(1) or m.group(2)], text)
    except KeyError:
        return None
    return None if ("$" in out or "`" in out) else out


def _weaker(a: str | None, b: str | None) -> str | None:
    return b if a is None else a if b is None else ";" if ";" in (a, b) else "&&"


def resolve_vars(command: str) -> str | None:
    """The command with simple shell variables resolved, for analysis only: `export VAR=v` / `VAR=v` steps are dropped and `$VAR` /
    `${VAR}` in later steps are replaced by the value; `VAR=v cmd` (prefix form) is `cmd` (the value is not visible to its own words).
    Steps are the `&&` / `;` pieces. None when it cannot be done safely: an unknown variable, `$(...)`, backquotes, `${VAR:-x}`, a value with
    whitespace / glob characters, an assignment to IFS / PATH / HOME ... A command without variables comes back unchanged."""
    if "$" not in command and not _ROUGH_ASSIGN.search(command):
        return command
    parts = _SEGMENT_SPLIT.split(command.strip())
    segments, seps = parts[0::2], parts[1::2]
    if not all(x.strip() for x in segments):
        return None
    env: dict[str, str] = {}
    kept: list[str] = []
    joins: list[str] = []
    link: str | None = None
    for n, seg in enumerate(segments):
        after = seps[n] if n < len(seps) else None
        try:
            toks = shlex.split(seg)
        except ValueError:
            return None
        exported = bool(toks) and toks[0] == "export"
        i = 1 if exported else 0
        assigns: list[tuple[str, str]] = []
        while i < len(toks) and _ASSIGN_TOK.match(toks[i]):
            name, _, raw = toks[i].partition("=")
            value = _subst(raw, env)
            if value is None or _BAD_VALUE.search(value) or name in _SHELL_VARS or name.startswith("LD_"):
                return None
            assigns.append((name, value))
            i += 1
        rest = toks[i:]
        if not toks or (exported and (rest or not assigns)):
            return None
        if assigns and not rest:
            env.update(assigns)
            link = _weaker(link, after)                    # the step is gone: what joined it to its neighbours is the weaker of the two
            continue
        new = [_subst(t, env) for t in rest]
        if any(t is None for t in new):
            return None
        if kept:
            joins.append(link or "&&")
        kept.append(seg.strip() if not assigns and new == rest else shlex.join([t for t in new if t is not None]))
        link = after
    return "".join(k + (f" {joins[n]} " if n < len(joins) else "") for n, k in enumerate(kept))


def _disposable(rel_parts: tuple[str, ...]) -> bool:
    return bool(rel_parts) and (rel_parts[0] in DISPOSABLE_DIRS or rel_parts[0].startswith((".tmp", "tmp_", ".scratch", "scratch_")))


def _tmp_ours(rel_parts: tuple[str, ...]) -> bool:
    """/tmp/<name>/...: only a folder that does not exist yet or belongs to this user (never /tmp/.X11-unix and friends)."""
    if not hasattr(os, "getuid"):                              # Windows has no /tmp and no uid: nothing under a shared temp folder is "ours"
        return False
    try:
        return os.lstat(_TMP.resolve() / rel_parts[0]).st_uid == os.getuid()
    except FileNotFoundError:
        return True
    except OSError:
        return False


def scratch_ok(command: str, root: Path, *, other_ok: Callable[[str], bool] | None = None, scratch_only: bool = False,
               owned: Mapping[str, str] | None = None, owned_origins: tuple[str, ...] = ("run", "write")) -> bool:
    """A file-shuffling command that only touches throwaway areas. The safety classifier rates every `rm` and every copy to
    /tmp as DANGEROUS, yet the same effect through `python -c os.remove()` passes - so a trusted (auto) repo could not even
    clean up its own test folder and the model burned steps working around the guard. Allowed here (and nothing else):
      * rm / rmdir only inside a disposable folder of the repo (.scratch, .tmp*, build, dist, __pycache__ ...) or under a
        /tmp/<name> of this user - a real repo file is removed with delete_file (recoverable), never with rm,
      * mkdir / touch inside the repo or such a /tmp/<name>,
      * cp / mv into a throwaway area, or to a NEW path in the repo (never over an existing file: no clobbering a save file);
        cp may read any repo file, mv may only move throwaway files (moving real files is move_file's job),
      * `cd` into an existing folder of those areas (later relative paths are resolved from there).
    Rejected outright: sudo, pipes, redirects, `$`, backquotes, backslashes, `&`, newlines, `..`, globs and braces, long
    options and unknown flags. Paths are resolved through symlinks before they are judged.
    Simple variables assigned earlier in the same command line (`export F=.scratch/x && rm -f $F`) are resolved first (resolve_vars);
    any other `$` stays rejected.
    `other_ok`: a command that is not a file verb (`python3 todo.py add x`) may follow the housekeeping when this says it is harmless; the
    file verbs must then all come BEFORE it (it could have changed what a later path points to, e.g. made a symlink).
    `scratch_only`: only rm / rmdir / mkdir, and only inside .scratch/ (what `medium` mode accepts without asking).
    `owned`: repo-relative file -> how the agent got it ("run": a command created it, "write": the model wrote it). A plain `rm` / `unlink`
    (no -r / -R) of such an existing regular file whose origin is in `owned_origins` counts as housekeeping too (the program under test
    leaves its data file behind and the next run starts by deleting it); a folder, a symlink, a glob or .git never does."""
    resolved = resolve_vars(command)
    if resolved is None:
        return False
    command = resolved
    if other_ok is None and _UNSAFE.search(command.replace("&&", " ")):
        return False
    root = root.resolve()
    tmp = _TMP.resolve()

    def classify(arg: str, base: Path) -> tuple[str, tuple[str, ...], Path] | None:
        if not arg or any(ch in arg for ch in "*?[]{}") or "~" in arg[1:] or (arg.startswith("~") and not arg.startswith("~/")):
            return None
        p = Path(os.path.expanduser(arg))
        p = p if p.is_absolute() else base / p
        try:
            p = p.resolve()
        except (OSError, RuntimeError):
            return None
        if root in p.parents:
            rel = p.relative_to(root).parts
            return None if ".git" in rel else ("repo", rel, p)
        if tmp in p.parents:
            return ("tmp", p.relative_to(tmp).parts, p)
        return None

    def throwaway(c: tuple[str, tuple[str, ...], Path]) -> bool:
        return _tmp_ours(c[1]) if c[0] == "tmp" else _disposable(c[1])

    def ours(c: tuple[str, tuple[str, ...], Path], verb: str, flags: list[str]) -> bool:
        if not owned or c[0] != "repo" or verb not in ("rm", "unlink") or any(f[1:] and set(f[1:]) & set("rR") for f in flags):
            return False
        return owned.get("/".join(c[1])) in owned_origins and c[2].is_file() and not c[2].is_symlink()

    def lands_safely(dst: str, sources: list[str], base: Path) -> bool:
        c = classify(dst, base)
        if c is None:
            return False
        if throwaway(c):
            return True
        if c[2].is_dir():                                  # cp a b dir/ -> dir/a, dir/b must be new
            return all(not (c[2] / Path(src).name).exists() for src in sources)
        return len(sources) == 1 and not c[2].exists()

    parts = _SEGMENT_SPLIT.split(command.strip())
    segments, seps = parts[0::2], parts[1::2]
    if not all(x.strip() for x in segments):
        return False
    base = root
    made: set[Path] = set()                                # folders an earlier `mkdir` of this command creates
    seen_other = False
    for n, seg in enumerate(segments):
        chained = all(sep == "&&" for sep in seps[:n])     # every earlier step must have succeeded to get here
        try:
            toks = shlex.split(seg)
        except ValueError:
            return False
        if not toks:
            return False
        if toks[0] not in _VERB_FLAGS and toks[0] != "cd" and other_ok is not None and not scratch_only:
            if not other_ok(seg):
                return False
            seen_other = True
            continue
        if seen_other or (other_ok is not None and _UNSAFE.search(seg)):
            return False
        if toks[0] == "cd":
            if scratch_only or len(toks) != 2 or toks[1].startswith("-"):
                return False
            target = Path(os.path.expanduser(toks[1]))
            target = (target if target.is_absolute() else base / target).resolve()
            if target != root and (classify(toks[1], base) is None or (tmp in target.parents and not _tmp_ours(target.relative_to(tmp).parts))):
                return False
            if not target.is_dir() and not (chained and target in made):     # a failed cd + `;` would run the rest elsewhere
                return False
            base = target
            continue
        verb = toks[0]
        if verb not in _VERB_FLAGS or (scratch_only and verb not in ("rm", "rmdir", "mkdir", "unlink")):
            return False
        flags = [t for t in toks[1:] if t.startswith("-")]
        args = [t for t in toks[1:] if not t.startswith("-")]
        if not args or any(not re.fullmatch(r"-[a-zA-Z]+", f) or set(f[1:]) - set(_VERB_FLAGS[verb]) for f in flags):
            return False
        classes = [classify(a, base) for a in args]
        if any(c is None for c in classes) or (scratch_only and not all(c[0] == "repo" and (c[1][:1] == (".scratch",) or ours(c, verb, flags)) for c in classes)):
            return False
        if verb in ("rm", "rmdir", "unlink"):
            if not all(throwaway(c) or ours(c, verb, flags) for c in classes):
                return False
            if verb == "unlink" and (len(args) != 1 or flags):
                return False
        elif verb in ("mkdir", "touch"):
            if any(c[0] == "tmp" and not _tmp_ours(c[1]) for c in classes):
                return False
            if verb == "mkdir":
                for c in classes:
                    made.update([c[2], *c[2].parents])
        else:                                              # cp / mv
            if len(args) < 2 or not lands_safely(args[-1], args[:-1], base):
                return False
            if verb == "mv" and not all(throwaway(c) for c in classes[:-1]):
                return False
    return True


def _not_dangerous(segment: str) -> bool:
    from agent.safety import RiskLevel, classify_command

    return classify_command(segment, {}) in (RiskLevel.SAFE, RiskLevel.CAUTION)


def check_command(pol: RepoPolicy, root: Path, command: str, risk_name: str, owned: Mapping[str, str] | None = None) -> Decision:
    """`risk_name` is the safety classifier's verdict (SAFE / CAUTION / DANGEROUS / BLOCKED). `owned`: the files the agent wrote or a run created
    (see scratch_ok): deleting one with a plain rm is housekeeping in auto mode, and in medium mode when a run created it."""
    if risk_name == "BLOCKED":
        return Decision("block", "blocked by the safety policy")
    hit = _matches(command, pol.deny)
    if hit:
        return Decision("block", f"matches your deny pattern {hit!r}")
    if not _root_in_temp(root) and TEMP_RE.search(command):
        return Decision("block", TEMP_REASON)
    why = forbidden_in_command(command, root)
    if why:
        return Decision("block", why)
    if risk_name == "DANGEROUS" and pol.mode == AUTO and scratch_ok(command, root, other_ok=_not_dangerous, owned=owned):
        risk_name = "CAUTION"                      # housekeeping inside a trusted repo / a throwaway folder (+ ordinary commands after it)
    elif risk_name == "DANGEROUS" and pol.mode == MEDIUM and scratch_ok(command, root, scratch_only=True, owned=owned, owned_origins=("run",)):
        return Decision("allow", "scratch housekeeping")
    if risk_name == "DANGEROUS":
        return Decision("ask", "dangerous command")
    if PUSH_RE.search(command):
        return Decision("allow" if _matches(command, pol.allow) else "ask", "publishes your work")
    if INSTALL_RE.search(command) and not pol.installs and not _matches(command, pol.allow):
        return Decision("ask", "installs packages (set installs=true or /allow it)")
    if NETWORK_RE.search(command) and not pol.network and not _matches(command, pol.allow):
        return Decision("ask", "uses the network (set network=true or /allow it)")
    if risk_name == "SAFE":
        return Decision("allow")
    if _matches(command, pol.allow):
        return Decision("allow", "matches your allow list")
    if pol.mode == AUTO:
        from agent import repoaccess

        return Decision("allow") if (repoaccess._shell_in_scope(command, root, cwd=root) or _scope_with_allow(command, root)) else Decision("ask", "reaches outside the repo")
    return Decision("ask", "commands need approval in this mode")


def describe(root: Path | str) -> str:
    pol = for_repo(root)
    lines = [f"repo   {Path(root).resolve()}",
             f"mode   {pol.mode}   commit {pol.commit}   network {'on' if pol.network else 'off'}   installs {'on' if pol.installs else 'off'}",
             f"steps  {pol.max_steps}   think {pol.think}" + (f"   model {pol.model}" if pol.model else "") + (f"   ctx {pol.ctx}" if pol.ctx else ""),
             f"allow  {', '.join(pol.allow) or '-'}", f"deny   {', '.join(pol.deny) or '-'}"]
    if pol.test_cmd:
        lines.append(f"tests  {pol.test_cmd}")
    return "\n".join(lines)


def parse_flag(text: str) -> bool | None:
    t = (text or "").strip().lower()
    return True if t in ("on", "true", "yes", "1") else False if t in ("off", "false", "no", "0") else None


def split_args(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()
