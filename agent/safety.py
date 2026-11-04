"""Risk classification for model-requested shell commands.

This is the security core. `classify_command` maps a shell command to one of
four risk tiers; the agent loop enforces an approval gate keyed on the tier.
The guiding rule is default-deny / strictest-match-wins: on ANY ambiguity we
escalate to the stricter tier, and a parse we cannot understand is DANGEROUS.

Config can only EXTEND the built-in allow/block lists (`safety.safe_commands`
and `safety.blocked_patterns`); it can never shrink them, so a misconfigured or
malicious config cannot make `rm -rf /` runnable.
"""

from __future__ import annotations

import contextvars
import re
import shlex
from enum import IntEnum
from typing import Any

from agent import shellparse


class RiskLevel(IntEnum):
    """Ordered so `max()` picks the strictest tier for compound commands."""

    SAFE = 0
    CAUTION = 1
    DANGEROUS = 2
    BLOCKED = 3


# Read-only executables that auto-run. Extended (never replaced) by config.
_BUILTIN_SAFE_COMMANDS = frozenset(
    {
        "ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "echo",
        "date", "whoami", "uname", "df", "du", "free", "ps", "top", "which",
        "file", "stat", "wc", "uptime",
        "test", "[", "true", "false",      # side-effect-free shell builtins
    }
)

# Multi-word read-only invocations. The bare executable (e.g. `git`) is NOT
# safe on its own (git push is dangerous), only these specific subcommands.
_BUILTIN_SAFE_PREFIXES = (
    ("git", "status"),
    ("git", "log"),
    ("git", "diff"),
)

# Escalate-to-DANGEROUS executables.
_DANGEROUS_EXECUTABLES = frozenset(
    {
        "sudo", "doas", "pkexec",
        "systemctl", "service",
        "apt", "apt-get", "dnf", "yum", "pacman", "snap",
        "kill", "pkill", "killall",
        # run a command as another user / in another root or namespace, or later (at/batch) / under a lock
        "su", "runuser", "chroot", "nsenter", "unshare", "setpriv", "at", "batch", "flock",
    }
)

# Privilege-elevation prefixes. Unlike wrappers these genuinely change
# semantics (run as root), so we do NOT strip them and re-classify the payload
# at the payload's own tier -- instead we FLOOR at DANGEROUS and additionally
# block-scan the payload, returning max(DANGEROUS, payload_tier). So `sudo ls`
# is DANGEROUS but `sudo rm -rf /` / `sudo reboot` stay BLOCKED.
_PRIV_ELEVATION = frozenset({"sudo", "doas", "pkexec"})

# sudo/doas/pkexec own value-taking options (spaced form consumes next token).
# Guarded by the same payload-eating logic as wrappers so `sudo -u root reboot`
# does not eat `reboot` as the `-u` value.
_PRIV_VALUE_OPTIONS = frozenset(
    {"-u", "-g", "-p", "-C", "-h", "-U", "-r", "-t"}
)

# Package managers whose install/remove verbs are dangerous (already covered by
# the executable set above, but kept explicit for readability of intent).
_PIPE_TO_SHELL = frozenset({"sh", "bash", "zsh", "dash", "tee"})

# Command-runner / wrapper prefixes. `wrapper rm -rf /` must be classified as
# though `rm -rf /` were the command: the wrapper cannot let the wrapped exe
# escape its own checks. After stripping the prefix (and its own trivial
# option/arg) we RE-CLASSIFY the remainder and take the max risk.
_WRAPPER_PREFIXES = frozenset(
    {
        "env", "command", "builtin", "exec", "nice", "nohup", "timeout",
        "stdbuf", "setsid", "xargs", "time", "ionice", "chrt", "taskset", "numactl",
    }
)

# For each known wrapper, the options that CONSUME A VALUE (whether numeric or a
# string like `KILL`/`best-effort`). Both short (`-s`) and long (`--signal`)
# forms are listed; the attached `-sKILL` / `--signal=KILL` forms need no extra
# token, the spaced `-s KILL` / `--signal KILL` forms consume the next token.
# A wrapper absent from this map (env, command, exec, nohup, time, setsid, ...)
# has no value-taking options of its own.
_WRAPPER_VALUE_OPTIONS = {
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "--class", "-n", "--classdata", "-p", "--pid"},
    "chrt": {"-p"},
    "stdbuf": {"-i", "-o", "-e"},
    "env": {"-u", "--unset", "-C", "--chdir", "-P", "-a", "--argv0"},
    "xargs": {"-a", "--arg-file", "-d", "--delimiter", "-E", "-I", "-L", "-n", "--max-args", "-P", "--max-procs",
              "-s", "--max-chars", "--process-slot-var"},
    "taskset": {"-p", "--pid"},
    "numactl": {"-C", "--physcpubind", "-N", "--cpunodebind", "-m", "--membind", "-i", "--interleave", "-p", "--preferred"},
}

# Commands that hand their arguments to a SHELL, which parses them again: `eval 'rm -rf ~'`, `watch 'rm -rf ~'`. Their arguments
# are classified as a command line of their own (joined, not quoted), with these own options first.
_SHELL_REPARSE = {"eval": frozenset(), "watch": frozenset({"-n", "--interval", "-q", "--equexit"})}

# Commands whose `-c STRING` (or `--command STRING`) runs STRING through a shell. su / runuser also switch user (DANGEROUS floor).
_DASH_C_RUNNERS = frozenset({"su", "runuser", "script"})

# `find` actions that delete, run programs or write files (`-fprint FILE` writes FILE).
_FIND_ACTIONS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"})

# Environment variables that make an otherwise read-only command run other programs or load other code:
# `GIT_EXTERNAL_DIFF=x git diff`, `PATH=. ls`, `LD_PRELOAD=x.so cat f`, `RIPGREP_CONFIG_PATH=f rg x` (the file can set --pre).
_RISKY_ENV = re.compile(r"(?:PATH|LD_\w+|DYLD_\w+|BASH_ENV|ENV|IFS|SHELLOPTS|BASHOPTS|PROMPT_COMMAND|PS4|GIT_\w+|PAGER|MANPAGER|"
                        r"LESSOPEN|LESSCLOSE|RIPGREP_CONFIG_PATH)=", re.S)

# A command word bash would still EXPAND (variable, substitution, glob, brace expansion) - its real name is unknown here.
_EXPANDING_WORD = re.compile(r"[$`*?\[]|\{[^}]*(?:,|\.\.)[^}]*\}")

# Command words that, appearing as an argument of an UNKNOWN command (`busybox rm -rf ~`, `taskset 1 reboot`), are very likely its
# payload: the rest of the line from that word on is classified as a command too (it can only make the result stricter).
_PAYLOAD_VERBS = frozenset({"rm", "dd", "shutdown", "reboot", "poweroff", "halt", "chmod", "chown", "mv", "cp", "kill", "pkill",
                            "killall", "systemctl", "service", "sudo", "doas", "pkexec", "sh", "bash", "zsh", "dash", "eval"})

# Output-redirect targets that are not files: discarding output or writing it to the terminal never changes anything.
_HARMLESS_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/fd/1", "/dev/fd/2"})
_OUTPUT_OPS = frozenset({">", ">>", ">|", "&>", "&>>", "<>", ">&", "&>|"})

# Splits a command line into segments run in sequence/parallel. We classify
# each segment independently and take the highest risk. This regex is only used
# as a cheap fallback; the primary splitter (`_split_segments`) is grouping- and
# quote-aware so separators inside `(...)`, `{...}`, or quotes are never split.
_SEGMENT_SEPARATORS = re.compile(r"&&|\|\||[;|]")

# Recursion depth cap: pathological nesting (of wrappers, groups, or command
# substitutions) hitting this cap is treated as DANGEROUS rather than looping.
_MAX_DEPTH = 25

# Sentinel returned by `_strip_wrapper_prefix` when it recognizes a wrapper but
# cannot confidently identify the wrapped payload (e.g. a value-taking option
# with no value to consume). The caller escalates to DANGEROUS -- never falls
# through to CAUTION with an unclassified payload.
_WRAPPER_UNCERTAIN = object()


def _strip_env_prefix(tokens: list[str]) -> list[str]:
    """Drop leading `VAR=value` environment assignments to expose the real exe."""
    out = list(tokens)
    while out and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", out[0]):
        out.pop(0)
    return out


def _split_segments(command: str) -> list[str]:
    """The simple commands of `command` (agent.shellparse: quote-, escape-, group-, comment- and here-document-aware; newline,
    `&`, `|&`, `;;` are separators like `;`, `&&`, `||`, `|`). For a line the parser rejects, the crude regex split, so a
    reason can still be found for an error message - classification itself treats such a line as DANGEROUS."""
    segs = shellparse.split(command)
    if segs is None:
        return [p.strip() for p in re.split(r"&&|\|\||[;|&\n]", command) if p.strip()]
    return [seg.text for seg in segs if seg.text]


def _has_command_substitution(command: str) -> bool:
    return "$(" in command or "`" in command or "<(" in command or ">(" in command


def _iter_redirects(text: str):
    """(operator, target) for every OUTPUT redirect in one simple command: `>`, `>>`, `>|`, `&>`, `&>>`, `<>` (read-write) and
    `>&FILE`. Operator-aware (a `>` glued to a word, `echo hi>~/.bashrc`, is still a redirect; one inside quotes is not). An fd
    duplication (`2>&1`, `>&2`, `>&-`) is not a file. The target is None when the line ends after the operator."""
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return
    for i, tok in enumerate(tokens):
        if tok not in _OUTPUT_OPS:
            continue
        target = tokens[i + 1] if i + 1 < len(tokens) else None
        if tok == ">&" and (target is None or re.fullmatch(r"\d+-?|-", target)):
            continue                                            # fd duplication / closing, not a file
        yield tok, target


def _has_output_redirect(command: str) -> bool:
    """True if the command writes a real file through a redirect (`/dev/null`, `/dev/stderr` ... are not files)."""
    for _op, target in _iter_redirects(command):
        if target is None or target.strip("'\"") not in _HARMLESS_TARGETS:
            return True
    return False


def _exe_basename(token: str) -> str:
    """Normalize an executable token to its basename for tier matching.

    A path-qualified (`/bin/rm`, `/sbin/reboot`) or `./`-relative (`./reboot`)
    invocation runs the same command as its bare basename, so the BLOCKED /
    DANGEROUS / command-word matchers must key off the basename or the payload
    slips every rule. Strips a leading escape backslash (`\\rm`) first, matching
    the existing bare-token handling.

    NOTE: this is deliberately NOT applied to the SAFE auto-run allowlist check.
    A path-qualified safe command (`/bin/ls`) must stay CAUTION (fail closed),
    not become SAFE.
    """
    return token.lstrip("\\").rsplit("/", 1)[-1]


def _raw_blocked_reason(command: str, blocked_patterns: list[str]) -> str | None:
    """The checks that look at the raw text (they must survive any tokenisation quirk): fork bomb, config patterns, block devices."""
    stripped = command.strip()
    collapsed = re.sub(r"\s+", "", stripped)

    # Fork bomb: match the classic `:(){:|:&};:` regardless of internal spacing.
    if ":(){" in collapsed and ":|:&" in collapsed:
        return "fork bomb"

    # Config-supplied substrings (extend the built-ins; case-sensitive match
    # against the raw command).
    for pattern in blocked_patterns:
        if pattern and pattern in command:
            return f"matches blocked pattern '{pattern}'"

    # Raw-text scans that must survive tokenisation quirks (attached redirects
    # like `echo hi>/dev/sda`, env-prefixed `DD=1 dd of=/dev/sda`, etc). We look
    # at the whole segment so nothing hides behind spacing or leading tokens.
    if re.search(r">>?\s*/dev/(sd|nvme|hd|vd|mmcblk)", stripped):
        return "redirect to block device"
    if re.search(r"\bof=\s*/dev/(sd|nvme|hd|vd|mmcblk)", stripped):
        return "dd to block device"
    return None


def _blocked_reason(command: str, blocked_patterns: list[str]) -> str | None:
    """Return a human reason if `command` (a single segment) is BLOCKED, else None."""
    raw = _raw_blocked_reason(command, blocked_patterns)
    if raw:
        return raw
    stripped = command.strip()

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None  # parse errors are handled by the caller as DANGEROUS
    if not tokens:
        return None

    # Skip leading `VAR=value` environment assignments to find the real exe.
    real = _strip_env_prefix(tokens)
    if not real:
        return None
    # A leading backslash escapes an alias but bash still runs the same exe
    # (`\rm` runs `rm`). Strip it before matching. A path-qualified exe
    # (`/bin/rm`, `./reboot`) runs the same command as its basename, so
    # normalize to the basename before BLOCKED matching -- otherwise a
    # path-qualified payload would slip every tier rule.
    exe = _exe_basename(real[0])
    args = real[1:]

    # --no-preserve-root anywhere is an unambiguous whole-disk wipe intent.
    if "--no-preserve-root" in args:
        return "--no-preserve-root"

    # mkfs / mkfs.* -> formatting a filesystem.
    if exe == "mkfs" or exe.startswith("mkfs."):
        return "mkfs (filesystem format)"

    # dd writing to a block device (token form; raw-text form handled above).
    if exe == "dd":
        for arg in args:
            if arg.startswith("of=") and _is_block_device(arg[len("of=") :]):
                return "dd to block device"

    # Every output redirect (`>`, `>>`, `&>`, `>|`, `<>`, `>&FILE`, glued or spaced): block devices and sensitive paths.
    for _op, target in _iter_redirects(stripped):
        if target and _is_block_device(target):
            return "redirect to block device"
        if target and _is_sensitive_redirect_target(target):
            return f"redirect to sensitive path {target}"

    # rm targeting root / home / glob-root.
    if exe == "rm":
        for arg in args:
            if _is_root_or_home_target(arg):
                return f"rm targeting {arg}"

    # chmod -R on root, home or system dir (recursive permission change on a
    # protected tree is catastrophic regardless of the mode).
    if exe == "chmod" and _is_recursive(args):
        for arg in args:
            if _is_root_or_home_target(arg):
                return "chmod -R on protected path"

    # System power state changes.
    if exe in {"shutdown", "reboot", "poweroff", "halt"}:
        return f"{exe} (power state change)"

    return None


def _is_block_device(path: str) -> bool:
    path = path.strip().strip("'\"")
    return bool(re.match(r"/dev/(sd|nvme|hd|vd|mmcblk)", path))


def _redirect_target(tokens: list[str]) -> str | None:
    """The file target of the first output redirect of an already tokenised command (None: there is none)."""
    for _op, target in _iter_redirects(" ".join(tokens)):
        if target:
            return target
    return None


# System directories whose recursive removal / world-write is catastrophic.
# rm -r* / chmod -R targeting any of these (or their `/*` globs) is BLOCKED.
_BLOCKED_SYSTEM_DIRS = frozenset(
    {
        "/home", "/etc", "/usr", "/boot", "/bin", "/lib", "/lib64",
        "/var", "/sys", "/proc", "/dev", "/sbin", "/root",
    }
)


def _is_root_or_home_target(arg: str) -> bool:
    arg = arg.strip().strip("'\"")
    if arg in {"/", "/*", "~", "~/", "$HOME", "$HOME/", "${HOME}"}:
        return True
    # System directories and their globs: `/etc`, `/etc/`, `/etc/*`.
    normalized = arg.rstrip("/")
    if normalized.endswith("/*"):
        normalized = normalized[:-2].rstrip("/")
    return normalized in _BLOCKED_SYSTEM_DIRS


def _is_sensitive_redirect_target(target: str) -> bool:
    """True if a `>`/`>>` target is an absolute path outside the launch cwd, or
    a home dotfile -- writing there escalates to at least DANGEROUS.
    """
    t = target.strip().strip("'\"")
    if not t:
        return False
    # /dev/null is a black hole, not a file - nothing written there can ever
    # be read back, leaked, or overwritten. `2>/dev/null`/`>/dev/null` to
    # discard noise is one of the most common, completely harmless shell
    # idioms there is; blocking it as "sensitive" forces the model to
    # abandon a perfectly safe command and reach for something worse.
    if t in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
        return False
    # Home dotfile: ~/.foo or $HOME/.foo (any hidden file under home).
    m = re.match(r"(?:~|\$HOME|\$\{HOME\})/?(\..+)", t)
    if m:
        return True
    # Absolute path (outside the project cwd -- we treat any absolute target as
    # outside, matching the conservative _is_path_outside_cwd stance).
    if t.startswith("/"):
        return True
    return False


def _is_recursive(args: list[str]) -> bool:
    for arg in args:
        if arg == "--recursive":
            return True
        # Bundled short flags like -Rf or -rf.
        if re.fullmatch(r"-[a-zA-Z]*[rR][a-zA-Z]*", arg):
            return True
    return False


def _is_path_outside_cwd(arg: str) -> bool:
    """A path is 'outside the project' if it is absolute or escapes upward.

    Home-relative (`~`) and parent-relative (`../`) both leave the CWD. Kept
    deliberately simple per the brief: mv/cp/rm on such paths -> DANGEROUS.
    """
    stripped = arg.strip("'\"")
    if not stripped or stripped.startswith("-"):
        return False
    if stripped.startswith("/"):
        return True
    if stripped.startswith("~") or stripped.startswith("$HOME") or stripped.startswith("${HOME}"):
        return True
    if stripped.startswith("../") or stripped == "..":
        return True
    return False


def _dangerous_reason(command: str) -> bool:
    """Return True if a single segment is DANGEROUS (assumes not BLOCKED)."""
    try:
        tokens = shlex.split(command.strip())
    except ValueError:
        return True  # shouldn't reach here (caller pre-checks) but be strict
    tokens = _strip_env_prefix(tokens)
    if not tokens:
        return False

    exe = _exe_basename(tokens[0])
    args = tokens[1:]

    # find with a destructive action (or one that writes a file: -fprint FILE) is not read-only despite `find` being on
    # the safe allowlist; treat it as DANGEROUS (strictest-wins on ambiguity).
    if exe == "find" and any(a in _FIND_ACTIONS for a in args):
        return True

    # ripgrep's --pre runs a program on every file it searches.
    if exe == "rg" and any(a == "--pre" or a.startswith("--pre=") for a in args):
        return True

    # `git diff/log --output=FILE` writes FILE; outside the project that is as risky as a redirect there.
    if exe == "git":
        for n, a in enumerate(args):
            if a == "--output" or a.startswith("--output="):
                target = a.split("=", 1)[1] if "=" in a else (args[n + 1] if n + 1 < len(args) else "")
                if not target or _is_sensitive_redirect_target(target):
                    return True

    # crontab installs (or removes) scheduled commands; only listing them is harmless.
    if exe == "crontab" and args != ["-l"]:
        return True

    # `date -s` / `date --set` changes the system clock.
    if exe == "date" and any(a in ("-s", "--set") or a.startswith(("--set=", "-s")) for a in args):
        return True

    if exe in _DANGEROUS_EXECUTABLES:
        return True

    # curl|sh-style pipe-to-shell: this segment IS the shell being fed. When a
    # pipe feeds a shell, the downstream segment's executable is sh/bash/etc.
    if exe in _PIPE_TO_SHELL:
        return True

    # rm -r / rm -f variants not caught as BLOCKED.
    if exe == "rm" and _is_recursive(args):
        return True
    if exe == "rm" and any(re.fullmatch(r"-[a-zA-Z]*f[a-zA-Z]*", a) for a in args):
        return True

    # dd (any remaining dd invocation).
    if exe == "dd":
        return True

    # git push --force / --force-with-lease.
    if exe == "git" and "push" in args and any(a.startswith("--force") for a in args):
        return True

    # Recursive chmod/chown.
    if exe in {"chmod", "chown"} and _is_recursive(args):
        return True

    # mv/cp/rm touching a path outside the project directory.
    if exe in {"mv", "cp", "rm"}:
        for arg in args:
            if _is_path_outside_cwd(arg):
                return True

    return False


# Executable names that are recognizable as command words (not option values).
# Union of every list a token could plausibly be a command from, so a payload
# like `rm`/`reboot`/`mkfs.ext4` is never mistaken for a bare option value such
# as `high`/`KILL`/`10`/`best-effort`.
_KNOWN_COMMAND_EXES = (
    _BUILTIN_SAFE_COMMANDS
    | _DANGEROUS_EXECUTABLES
    | _PIPE_TO_SHELL
    | _WRAPPER_PREFIXES
    | frozenset(
        {
            "rm", "mv", "cp", "chmod", "chown", "dd", "shutdown", "reboot",
            "poweroff", "halt", "mkfs", "git",
        }
    )
)


def _looks_like_command_not_value(token: str) -> bool:
    """True if `token` looks like a payload command word rather than a plausible
    bare option value (`high`, `KILL`, `10`, `best-effort`).

    Used to refuse eating the real payload as a value-option's value. A token is
    treated as a command (not a value) when it is a known executable, an `mkfs.*`
    variant, or a backslash-escaped exe (`\\rm`). Bare values are unaffected.
    """
    t = _exe_basename(token)
    if t in _KNOWN_COMMAND_EXES:
        return True
    if t == "mkfs" or t.startswith("mkfs."):
        return True
    return False


def _strip_wrapper_prefix(tokens: list[str]) -> list[str] | None | object:
    """If `tokens` begins with a known command-runner prefix, strip it (and its
    own trivial option/arg) and return the remainder to be RE-classified.

    Returns None if tokens[0] is not a wrapper prefix. Returns a possibly-empty
    list otherwise. The caller re-classifies the remainder and takes the max.
    We consume the wrapper's own leading `-x` options and, for wrappers whose
    first positional is their own arg (timeout DURATION, nice -n N handled by
    option consumption, chrt PRIO), the leading non-command positional.
    """
    if not tokens:
        return None
    exe = tokens[0].lstrip("\\")
    if exe not in _WRAPPER_PREFIXES:
        return None
    rest = tokens[1:]
    value_opts = _WRAPPER_VALUE_OPTIONS.get(exe, frozenset())

    # Consume the wrapper's own leading options. For an option known to take a
    # value we consume its value regardless of whether it is numeric or a string
    # (`-s KILL`, `-c best-effort`, `-n high`) -- otherwise the value would be
    # mis-read as the payload executable, hiding the real (possibly BLOCKED)
    # command behind the wrapper.
    while rest and rest[0].startswith("-"):
        opt = rest.pop(0)
        if opt == "--":
            break  # end-of-options marker; everything after is the payload
        # `env -S STRING` / `--split-string=STRING` runs STRING split into words: STRING is the payload.
        if exe == "env" and (opt in ("-S", "--split-string") or opt.startswith("--split-string=")
                             or (opt.startswith("-S") and len(opt) > 2)):
            if opt in ("-S", "--split-string"):
                if not rest:
                    return _WRAPPER_UNCERTAIN
                value = rest.pop(0)
            else:
                value = opt.split("=", 1)[1] if opt.startswith("--") else opt[2:]
            try:
                return shlex.split(value) + rest
            except ValueError:
                return _WRAPPER_UNCERTAIN
        # Attached-value long form `--signal=KILL`: the value is glued on, so no
        # extra token to consume regardless of whether we recognize the option.
        if opt.startswith("--") and "=" in opt:
            continue
        # Bare long form `--signal KILL`: consume the next token iff it's a
        # value-taking option we know AND the next token is a plausible value
        # (not the payload command). If the "value" is actually the payload
        # (`--signal rm ...`), leave it so the payload is reclassified.
        if opt.startswith("--"):
            if opt in value_opts:
                if not rest:
                    return _WRAPPER_UNCERTAIN  # value-option with no value: unsure
                if _looks_like_command_not_value(rest[0]):
                    # The "value" is actually the payload command (`--signal rm`):
                    # leave it in `rest` so the real (possibly BLOCKED) payload is
                    # reclassified rather than eaten and dropped.
                    return rest
                if rest[0].startswith("-"):
                    return _WRAPPER_UNCERTAIN  # ambiguous next option -> escalate
                rest.pop(0)
            continue
        # Bare short option `-s` that takes a value (`-s KILL`): consume next tok,
        # same payload-eating guard as the long form above.
        if opt in value_opts:
            if not rest:
                return _WRAPPER_UNCERTAIN
            if _looks_like_command_not_value(rest[0]):
                return rest  # payload, not a value: reclassify it
            if rest[0].startswith("-"):
                return _WRAPPER_UNCERTAIN
            rest.pop(0)
            continue
        # Attached-value short option `-sKILL` / `-n10`: `-s` head takes a value
        # and the value is glued on -- nothing more to consume.
        if opt[:2] in value_opts and len(opt) > 2:
            continue
        # Plain valueless short flag(s) (`-p` for chrt, or an unknown `-x`).
        continue

    # timeout / chrt take a leading positional (DURATION / PRIORITY) that is not
    # a command. Consume one leading numeric/duration token for these.
    if exe in {"timeout", "chrt"} and rest and re.fullmatch(r"\d+[a-zA-Z]?", rest[0]):
        rest.pop(0)
    # taskset MASK / taskset -c CPU-LIST: the first positional is the CPU set.
    if exe == "taskset" and rest and re.fullmatch(r"[0-9a-fA-Fx,\-]+", rest[0]):
        rest.pop(0)
    return rest


def _strip_priv_prefix(tokens: list[str]) -> list[str] | object | None:
    """For `sudo`/`doas`/`pkexec`, drop the elevation exe and its OWN options
    (including value-taking options like `-u user`) and return the payload
    tokens to be block-scanned. Returns None if tokens[0] is not an elevation
    prefix; `_WRAPPER_UNCERTAIN` if an option genuinely ate an ambiguous value.

    Same payload-eating guard as `_strip_wrapper_prefix`: a value-option whose
    "value" is actually a command word (`sudo -u root reboot` where `-u`'s value
    is `root`, fine; but a missing value must not swallow the payload) is left in
    place so the real payload is seen.
    """
    if not tokens:
        return None
    exe = tokens[0].lstrip("\\")
    if exe not in _PRIV_ELEVATION:
        return None
    rest = tokens[1:]
    while rest and rest[0].startswith("-"):
        opt = rest.pop(0)
        if opt == "--":
            break
        if opt.startswith("--") and "=" in opt:
            continue
        # Bare short value-option (`-u root`): consume its value unless the next
        # token is actually the payload command (missing value case).
        if opt in _PRIV_VALUE_OPTIONS:
            if not rest:
                return _WRAPPER_UNCERTAIN
            if _looks_like_command_not_value(rest[0]):
                return rest  # payload, not a value
            if rest[0].startswith("-"):
                return _WRAPPER_UNCERTAIN
            rest.pop(0)
            continue
        # Valueless flag (`-E`, `-H`, `-S`, ...) or unknown short option.
        continue
    return rest


def _extract_substitutions(command: str) -> list[str]:
    """Return the inner command strings of every `$(...)`, `<(...)`, `>(...)` and backtick span.

    Handles nested parentheses by balanced scanning. Backtick spans are treated as flat (non-nesting, per POSIX). Quotes are
    deliberately NOT honoured (a `$(...)` inside "..." runs; one inside '...' is classified too - stricter, never looser). Used so
    substituted commands are classified through the SAME path as the outer command.
    """
    inners: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        if command[i] in "$<>" and i + 1 < n and command[i + 1] == "(":
            depth = 1
            j = i + 2
            start = j
            while j < n and depth > 0:
                if command[j] == "(":
                    depth += 1
                elif command[j] == ")":
                    depth -= 1
                if depth == 0:
                    break
                j += 1
            inners.append(command[start:j])
            i = j + 1
        elif command[i] == "`":
            j = command.find("`", i + 1)
            if j == -1:
                break
            inners.append(command[i + 1 : j])
            i = j + 1
        else:
            i += 1
    return inners


def _strip_grouping(segment: str) -> str | None:
    """If `segment` is exactly a `(...)` subshell or a `{ ...; }` group, return the inner command list to be reclassified, else None.
    A brace only opens a group when a blank follows it (`{rm,-rf,~}` is brace EXPANSION, not a group)."""
    span = shellparse.group_span(segment)
    if span is None or span[1].strip():
        return None
    return span[0]


def _grouping_unbalanced(segment: str) -> bool:
    """True if the segment starts a `(` subshell or a `{ ` group that never closes."""
    s = segment.strip()
    opens = s.startswith("(") or (s.startswith("{") and len(s) > 1 and s[1] in " \t\n")
    return opens and shellparse.group_span(s) is None


# Reserved words that only introduce the real command (`if rm -rf ~; then`, `do reboot`, `! rm -rf /`) and words that close a
# compound command and run nothing themselves.
_LEADING_KEYWORDS = re.compile(r"(?:if|then|else|elif|do|while|until|!)(?=\s|$)\s*")
_CLOSING_WORDS = frozenset({"fi", "done", "esac", "then", "else", "do"})
_UNANALYSED_COMPOUNDS = re.compile(r"(?:case|select|coproc)(?=\s|$)")
_FOR_HEADER = re.compile(r"for\s+[A-Za-z_]\w*(?:\s+in(?=\s|$)(.*))?$", re.S)
_ARITH_FOR = re.compile(r"for\s*\(\(.*\)\)$", re.S)
_FUNCTION_DEF = re.compile(r"(?:function\s+)?([A-Za-z_:][\w.:-]*)\s*\(\s*\)\s*(.*)$|function\s+([A-Za-z_:][\w.:-]*)\s+(.*)$", re.S)
_HEAD_VARIABLE = re.compile(r'\s*"?\$(?:\{([A-Za-z_]\w*)\}|([A-Za-z_]\w*))"?(?=\s|$)')
_ASSIGN_TOKEN = re.compile(r"([A-Za-z_]\w*)=(.*)", re.S)
_UNSAFE_VALUE = re.compile(r"[\s$`*?\[\]{}~;&|<>()'\"\\]")

# Collects the BLOCKED reasons found during one classification (see `block_reason`): one code path decides AND explains.
_REASONS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("safety_reasons", default=None)


def _strip_keywords(seg: str) -> str | None:
    """`seg` without leading reserved words (`if`, `then`, `do`, `while`, `!` ...). None for `case`/`select`/`coproc`, whose
    bodies are not analysed here (the caller treats them as DANGEROUS)."""
    s = seg.strip()
    while True:
        if _UNANALYSED_COMPOUNDS.match(s):
            return None
        m = _LEADING_KEYWORDS.match(s)
        if not m or not m.group(0):
            return s
        s = s[m.end():].strip()


def _function_body(seg: str) -> str | None:
    """The body of a function definition (`name() { ...; }`, `function name { ...; }`), or None."""
    m = _FUNCTION_DEF.match(seg.strip())
    if not m:
        return None
    body = m.group(2) if m.group(1) is not None else m.group(4)
    return (body or "").strip()


def _resolve_head_variable(text: str, env: dict[str, str]) -> str:
    """`x=rm; $x -rf ~`: a command word that is a variable set earlier IN THIS LINE to a plain value is replaced by that value,
    so the real command is classified. Anything else is left alone (an unknown `$x` as command word is DANGEROUS later)."""
    m = _HEAD_VARIABLE.match(text)
    if not m:
        return text
    name = m.group(1) or m.group(2)
    if name not in env:
        return text
    return shlex.quote(env[name]) + text[m.end():]


def _remember_assignments(text: str, env: dict[str, str]) -> None:
    """Record the plain `NAME=value` assignments of a segment that consists of nothing else (they persist for later segments)."""
    try:
        toks = shlex.split(text)
    except ValueError:
        return
    if toks and toks[0] in ("export", "readonly", "declare", "typeset", "local"):
        toks = [t for t in toks[1:] if not t.startswith("-")]
    if not toks or not all(_ASSIGN_TOKEN.fullmatch(t) for t in toks):
        return
    for t in toks:
        m = _ASSIGN_TOKEN.fullmatch(t)
        if m is None:
            continue
        name, value = m.group(1), m.group(2)
        if not value or _UNSAFE_VALUE.search(value):
            env.pop(name, None)
        else:
            env[name] = value


def _note_blocked(reason: str) -> None:
    found = _REASONS.get()
    if found is not None and reason not in found:
        found.append(reason)


def _classify_segment(
    segment: str,
    safe_commands: frozenset[str],
    blocked_patterns: list[str],
    depth: int,
) -> RiskLevel:
    """Classify a single (already segment-split) command.

    Recurses into groups, function bodies, wrapper prefixes, shell re-parsers (`eval`, `watch`, `su -c` ...) and command
    substitutions, running every extracted sub-command through the SAME path. `depth` guards against pathological nesting:
    hitting the cap is DANGEROUS.
    """
    if depth > _MAX_DEPTH:
        return RiskLevel.DANGEROUS

    seg = segment.strip()
    if not seg:
        return RiskLevel.SAFE

    # BLOCKED check first: raw-text + token scans (backslash/env-prefix aware).
    reason = _blocked_reason(seg, blocked_patterns)
    if reason is not None:
        _note_blocked(reason)
        return RiskLevel.BLOCKED

    # Reserved words only introduce the real command: `then rm -rf ~` is `rm -rf ~`.
    stripped = _strip_keywords(seg)
    if stripped is None:
        return RiskLevel.DANGEROUS          # case / select / coproc: not analysed, fail closed
    if stripped != seg:
        if not stripped or stripped in _CLOSING_WORDS:
            return RiskLevel.SAFE
        return _classify_segment(stripped, safe_commands, blocked_patterns, depth + 1)
    if seg in _CLOSING_WORDS:
        return RiskLevel.SAFE

    # `for NAME in WORDS`: only WORDS can do anything (substitutions), exactly like `echo WORDS`.
    m = _FOR_HEADER.match(seg)
    if m:
        return _classify_segment("echo " + (m.group(1) or ""), safe_commands, blocked_patterns, depth + 1)
    if _ARITH_FOR.match(seg):
        return RiskLevel.CAUTION

    # A function definition: its body is code (classified), and defining a function is at least CAUTION (it can shadow `ls`).
    body = _function_body(seg)
    if body is not None:
        return max(RiskLevel.CAUTION, _classify(body, safe_commands, blocked_patterns, depth + 1)) if body else RiskLevel.CAUTION

    # Subshell `(...)` / group `{ ...; }`, possibly followed by a redirect: reclassify the inner content (which may itself be
    # compound) through the full recursive path, and the rest as the redirect of a harmless `echo`.
    if _grouping_unbalanced(seg):
        return RiskLevel.DANGEROUS
    span = shellparse.group_span(seg)
    if span is not None:
        inner, rest = span
        risk = _classify(inner, safe_commands, blocked_patterns, depth + 1)
        if rest.strip():
            risk = max(risk, _classify_segment("echo " + rest.strip(), safe_commands, blocked_patterns, depth + 1))
        return risk

    # Command substitutions `$(...)` / `<(...)` / backticks: classify each inner command and fold into the max, then continue
    # classifying the outer command with the substitutions still present (they demote SAFE below).
    worst = RiskLevel.SAFE
    for inner in _extract_substitutions(seg):
        worst = max(worst, _classify(inner, safe_commands, blocked_patterns, depth + 1))

    # A standalone substitution (`$(rm -rf /)`) has no outer executable; its
    # risk is entirely the inner command's, already folded above.
    stripped_of_subs = re.sub(r"[$<>]\([^)]*\)|`[^`]*`", "", seg).strip()
    if not stripped_of_subs:
        return worst

    # A substitution or a file redirect anywhere in the segment means it is never SAFE, whatever wrapper comes first
    # (`env echo hi > out.txt` must not lose its redirect when the payload is re-quoted below).
    if _has_command_substitution(seg) or _has_output_redirect(seg):
        worst = max(worst, RiskLevel.CAUTION)

    try:
        tokens = shlex.split(seg)
    except ValueError:
        return RiskLevel.DANGEROUS
    real = _strip_env_prefix(tokens)
    if any(_RISKY_ENV.match(t) for t in tokens[: len(tokens) - len(real)]):
        worst = max(worst, RiskLevel.DANGEROUS)      # PATH=. ls / GIT_EXTERNAL_DIFF=x git diff / LD_PRELOAD=x cat f
    tokens = real
    if not tokens:
        return worst

    # A command word bash still expands (`$x`, `$(...)rm`, `/bin/r?`, `{rm,-rf,~}`, `$'\x72m'`): its real name is unknown.
    head = tokens[0].lstrip("\\")
    if head not in ("[", "[[") and _EXPANDING_WORD.search(head):
        return max(worst, RiskLevel.DANGEROUS)
    exe = _exe_basename(tokens[0])

    # Shell re-parsers: `eval STRING`, `watch STRING` run their joined arguments as a command line.
    if exe in _SHELL_REPARSE:
        rest = tokens[1:]
        own = _SHELL_REPARSE[exe]
        while rest and rest[0].startswith("-"):
            opt = rest.pop(0)
            if opt == "--":
                break
            if opt in own and rest:
                rest.pop(0)
        risk = max(worst, RiskLevel.CAUTION)
        if rest:
            risk = max(risk, _classify(" ".join(rest), safe_commands, blocked_patterns, depth + 1))
        return risk

    # `su -c STRING`, `runuser -c STRING`, `script -c STRING`: STRING runs through a shell (su/runuser also switch user).
    if exe in _DASH_C_RUNNERS:
        risk = max(worst, RiskLevel.DANGEROUS if exe in ("su", "runuser") else RiskLevel.CAUTION)
        for n, t in enumerate(tokens[1:], 1):
            payload = None
            if t in ("-c", "--command") and n + 1 < len(tokens):
                payload = tokens[n + 1]
            elif t.startswith("--command="):
                payload = t.split("=", 1)[1]
            if payload:
                risk = max(risk, _classify(payload, safe_commands, blocked_patterns, depth + 1))
                break
        return risk

    # Privilege elevation (`sudo`/`doas`/`pkexec`): floor at DANGEROUS but also
    # block-scan the payload so a blocked verb behind sudo stays BLOCKED. sudo
    # is NOT stripped-and-reclassified at the payload's own tier (it genuinely
    # elevates), only its BLOCKED-ness is propagated up.
    priv_payload = _strip_priv_prefix(tokens)
    if priv_payload is _WRAPPER_UNCERTAIN:
        return max(worst, RiskLevel.DANGEROUS)
    if priv_payload is not None:
        assert isinstance(priv_payload, list)
        floor = max(worst, RiskLevel.DANGEROUS)
        if not priv_payload:
            return floor  # `sudo` alone -> DANGEROUS
        inner_cmd = " ".join(shlex.quote(t) for t in priv_payload)
        payload_tier = _classify(inner_cmd, safe_commands, blocked_patterns, depth + 1)
        # Only propagate BLOCKED upward; a merely-CAUTION/SAFE payload does not
        # lower sudo's DANGEROUS floor, and its DANGEROUS is already the floor.
        return max(floor, payload_tier)

    remainder = _strip_wrapper_prefix(tokens)
    if remainder is _WRAPPER_UNCERTAIN:
        # Recognized a wrapper but could not identify the payload -> escalate.
        return max(worst, RiskLevel.DANGEROUS)
    if remainder is not None:
        assert isinstance(remainder, list)
        if not remainder:
            # `env` with nothing after it is just an env dump -> SAFE-ish.
            return worst
        inner_cmd = " ".join(shlex.quote(t) for t in remainder)
        return max(worst, _classify(inner_cmd, safe_commands, blocked_patterns, depth + 1))

    if _dangerous_reason(seg):
        return max(worst, RiskLevel.DANGEROUS)

    # SAFE requires: allowlisted exe (bare name, not path-qualified) AND no redirect AND no substitution (folded into `worst`).
    bare = tokens[0].lstrip("\\")
    if bare in safe_commands or (len(tokens) >= 2 and (bare, tokens[1]) in _BUILTIN_SAFE_PREFIXES):
        return worst

    # An unknown command: a known destructive command word among its arguments (`busybox rm -rf ~`, `taskset 1 reboot`,
    # `docker exec c rm -rf /x`) is very likely its payload, so the rest of the line from there is classified too.
    risk = max(worst, RiskLevel.CAUTION)
    for k in range(1, len(tokens)):
        word = _exe_basename(tokens[k])
        if word in _PAYLOAD_VERBS or word == "mkfs" or word.startswith("mkfs."):
            tail = " ".join(shlex.quote(t) for t in tokens[k:])
            risk = max(risk, _classify(tail, safe_commands, blocked_patterns, depth + 1))
            break
    return risk


def _classify(
    command: str,
    safe_commands: frozenset[str],
    blocked_patterns: list[str],
    depth: int,
) -> RiskLevel:
    """Segment-split `command` and classify each segment; strictest wins."""
    if depth > _MAX_DEPTH:
        return RiskLevel.DANGEROUS
    if not command or not command.strip():
        return RiskLevel.SAFE

    # Split into simple commands FIRST (agent.shellparse: quote/escape/group/heredoc aware; newline and `&` separate too).
    # Only then does each segment strip its own grouping and recurse. A line the parser rejects is DANGEROUS.
    segments = shellparse.split(command)
    if segments is None:
        return RiskLevel.DANGEROUS

    worst = RiskLevel.SAFE
    env: dict[str, str] = {}
    for segment in segments:
        text = _resolve_head_variable(segment.text, env)
        worst = max(worst, _classify_segment(text, safe_commands, blocked_patterns, depth + 1))
        # The body of an unquoted here-document is expanded: its `$(...)` / backquotes run.
        for doc in segment.heredocs:
            if doc.expands:
                for inner in _extract_substitutions(doc.body):
                    worst = max(worst, _classify(inner, safe_commands, blocked_patterns, depth + 1))
        _remember_assignments(segment.text, env)
    return worst


def classify_command(command: str, config: dict[str, Any]) -> RiskLevel:
    """Classify a shell command into a RiskLevel.

    Compound commands (`;`, `&&`, `||`, `|`, `|&`, `&`, newline) are split and every segment is classified; the highest
    (strictest) risk wins. Wrapper prefixes, subshell groups, reserved words, function bodies, shell re-parsers and command
    substitutions are recursively descended so the wrapped/grouped/substituted command carries its own risk (a wrapper can
    never let a blocked exe escape). A line the shell parser rejects is DANGEROUS. Config extends the built-in safe/blocked
    lists.
    """
    safety = config.get("safety", {}) if config else {}
    blocked_patterns = list(safety.get("blocked_patterns") or [])
    safe_commands = _BUILTIN_SAFE_COMMANDS | frozenset(safety.get("safe_commands") or [])

    if not command or not command.strip():
        return RiskLevel.SAFE

    # Belt-and-suspenders: the raw-text checks (fork bomb, config patterns, block devices) see the WHOLE line, here-document
    # bodies included, so nothing hides behind the way the line was split.
    raw = _raw_blocked_reason(command, blocked_patterns)
    if raw is not None:
        _note_blocked(raw)
        return RiskLevel.BLOCKED

    # A parse failure on the whole command means we cannot reason about it.
    if shellparse.split(command) is None:
        return RiskLevel.DANGEROUS

    return _classify(command, safe_commands, blocked_patterns, 0)


def block_reason(command: str, config: dict[str, Any]) -> str:
    """Return the human-readable reason a command is BLOCKED (for tool result): the first reason the classifier itself found,
    so the explanation can never drift from the decision."""
    found: list[str] = []
    token = _REASONS.set(found)
    try:
        classify_command(command, config)
    finally:
        _REASONS.reset(token)
    return found[0] if found else "blocked command"
