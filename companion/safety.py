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

import re
import shlex
from enum import IntEnum
from typing import Any


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
        "stdbuf", "setsid", "xargs", "time", "ionice", "chrt",
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
}

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
    """Split on compound separators (`;`, `&&`, `||`, `|`), grouping- and
    quote-aware.

    A separator is only a top-level separator when it sits OUTSIDE any `(...)`
    subshell, `{...}` group, and outside single/double quotes. A separator
    nested inside a group or quotes is part of that group/argument and is not a
    split point. Redirect/fd-dup pipes are handled by the caller; here `|` is
    always a separator when unquoted/ungrouped (a bare `|` is a pipe).
    """
    segments: list[str] = []
    buf: list[str] = []
    paren = 0  # depth of ( )
    brace = 0  # depth of { }
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        c = command[i]
        if quote is not None:
            buf.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "(":
            paren += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            if paren > 0:
                paren -= 1
            buf.append(c)
            i += 1
            continue
        if c == "{":
            brace += 1
            buf.append(c)
            i += 1
            continue
        if c == "}":
            if brace > 0:
                brace -= 1
            buf.append(c)
            i += 1
            continue
        if paren == 0 and brace == 0:
            # Two-char separators first.
            if command.startswith("&&", i) or command.startswith("||", i):
                segments.append("".join(buf))
                buf = []
                i += 2
                continue
            if c in ";|":
                segments.append("".join(buf))
                buf = []
                i += 1
                continue
        buf.append(c)
        i += 1
    segments.append("".join(buf))
    return [seg.strip() for seg in segments if seg.strip()]


def _has_command_substitution(command: str) -> bool:
    return "$(" in command or "`" in command


def _iter_redirect_tokens(tokens: list[str]):
    """Yield redirect operator/target info from tokenised output.

    A redirect is a `>`/`>>` operator (possibly with a leading fd digit like
    `2>`, or attached target like `>file`). A pure fd-dup (`2>&1`, `>&2`) is NOT
    a file redirect. Tokens come from shlex, so a `>` inside a quoted argument
    is already glued into its token and won't be seen here as an operator.
    """
    for i, tok in enumerate(tokens):
        m = re.fullmatch(r"(\d*)(>>?)(&?)(.*)", tok)
        if not m:
            continue
        _, op, amp, rest = m.groups()
        if amp:
            continue  # fd-dup like 2>&1 or >&2 -- not a file redirect
        if rest:
            yield op, rest
        elif i + 1 < len(tokens):
            yield op, tokens[i + 1]
        else:
            yield op, None


def _has_output_redirect(command: str) -> bool:
    """True if the command has a real `>`/`>>` file redirect (shlex-aware).

    A `>` inside a quoted argument (`echo 'a > b'`) or an fd-dup (`2>&1`) is
    NOT a redirect. Parse errors fall back to the conservative substring check.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ">" in command
    return any(True for _ in _iter_redirect_tokens(tokens))


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


def _blocked_reason(command: str, blocked_patterns: list[str]) -> str | None:
    """Return a human reason if `command` (a single segment) is BLOCKED, else None."""
    stripped = command.strip()
    lowered = stripped.lower()
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

    # Redirect (> / >>) whose target is a block device (spaced token form).
    redirect_target = _redirect_target(tokens)
    if redirect_target and _is_block_device(redirect_target):
        return "redirect to block device"

    # rm targeting root / home / glob-root.
    if exe == "rm":
        for arg in args:
            if _is_root_or_home_target(arg):
                return f"rm targeting {arg}"

    # chmod -R on root/home/system dir (recursive permission change on a
    # protected tree is catastrophic regardless of the mode).
    if exe == "chmod" and _is_recursive(args):
        for arg in args:
            if _is_root_or_home_target(arg):
                return "chmod -R on protected path"

    # Redirect (> / >>) to a sensitive path (absolute outside cwd, or home
    # dotfile). Block-device targets already handled above.
    if redirect_target and _is_sensitive_redirect_target(redirect_target):
        return f"redirect to sensitive path {redirect_target}"

    # System power state changes.
    if exe in {"shutdown", "reboot", "poweroff", "halt"}:
        return f"{exe} (power state change)"

    return None


def _is_block_device(path: str) -> bool:
    path = path.strip().strip("'\"")
    return bool(re.match(r"/dev/(sd|nvme|hd|vd|mmcblk)", path))


def _redirect_target(tokens: list[str]) -> str | None:
    """Return the file target of a `>`/`>>` redirect, if any."""
    for _op, target in _iter_redirect_tokens(tokens):
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

    # find with a destructive action is not read-only despite `find` being on
    # the safe allowlist; treat it as DANGEROUS (strictest-wins on ambiguity).
    if exe == "find" and any(a in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for a in args):
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
    """Return the inner command strings of every `$(...)` and backtick span.

    Handles nested `$(...)` by balanced-paren scanning. Backtick spans are
    treated as flat (non-nesting, per POSIX). Used so substituted commands are
    classified through the SAME path as the outer command.
    """
    inners: list[str] = []
    i = 0
    n = len(command)
    while i < n:
        if command[i] == "$" and i + 1 < n and command[i + 1] == "(":
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
    """If `segment` is a `(...)` subshell or `{ ...; }` group, return the inner
    content to be reclassified. Returns None if not a group.

    Unbalanced grouping is signalled by returning the sentinel-free path: the
    caller checks balance separately and escalates.
    """
    s = segment.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1].strip()
    if s.startswith("{") and s.endswith("}"):
        return s[1:-1].strip().rstrip(";").strip()
    return None


def _grouping_unbalanced(segment: str) -> bool:
    """True if grouping parens/braces are unbalanced at the top of a segment."""
    s = segment.strip()
    if s.startswith("(") and not s.endswith(")"):
        return True
    if s.startswith("{") and not s.endswith("}"):
        return True
    return False


def _classify_segment(
    segment: str,
    safe_commands: frozenset[str],
    blocked_patterns: list[str],
    depth: int,
) -> RiskLevel:
    """Classify a single (already segment-split) command.

    Recurses into groups, wrapper prefixes, and command substitutions, running
    every extracted sub-command through the SAME path. `depth` guards against
    pathological nesting: hitting the cap is DANGEROUS.
    """
    if depth > _MAX_DEPTH:
        return RiskLevel.DANGEROUS

    seg = segment.strip()
    if not seg:
        return RiskLevel.SAFE

    # BLOCKED check first: raw-text + token scans (backslash/env-prefix aware).
    if _blocked_reason(seg, blocked_patterns) is not None:
        return RiskLevel.BLOCKED

    # Unbalanced grouping: cannot reason about it -> DANGEROUS.
    if _grouping_unbalanced(seg):
        return RiskLevel.DANGEROUS

    # Subshell `(...)` / group `{ ...; }`: reclassify the inner content (which
    # may itself be compound) through the full recursive path.
    inner_group = _strip_grouping(seg)
    if inner_group is not None:
        return _classify(inner_group, safe_commands, blocked_patterns, depth + 1)

    # Command substitutions `$(...)` / backticks: classify each inner command
    # and fold into the max, then continue classifying the outer command with
    # the substitutions still present (they demote SAFE below).
    worst = RiskLevel.SAFE
    for inner in _extract_substitutions(seg):
        worst = max(worst, _classify(inner, safe_commands, blocked_patterns, depth + 1))

    # A standalone substitution (`$(rm -rf /)`) has no outer executable; its
    # risk is entirely the inner command's, already folded above.
    stripped_of_subs = re.sub(r"\$\([^)]*\)|`[^`]*`", "", seg).strip()
    if not stripped_of_subs:
        return worst

    # Wrapper / command-runner prefix (`env`, `timeout 5`, `nice`, ...): strip
    # it and RE-CLASSIFY the remainder, taking the max. The wrapper itself is
    # not risky; the wrapped command carries the risk.
    try:
        tokens = shlex.split(seg)
    except ValueError:
        return RiskLevel.DANGEROUS
    tokens = _strip_env_prefix(tokens)
    if not tokens:
        return max(worst, RiskLevel.SAFE)

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
            return max(worst, RiskLevel.SAFE)
        inner_cmd = " ".join(shlex.quote(t) for t in remainder)
        return max(worst, _classify(inner_cmd, safe_commands, blocked_patterns, depth + 1))

    # SAFE requires: allowlisted exe AND no redirect AND no substitution.
    if worst >= RiskLevel.CAUTION or _has_command_substitution(seg) or _has_output_redirect(seg):
        if _dangerous_reason(seg):
            return max(worst, RiskLevel.DANGEROUS)
        return max(worst, RiskLevel.CAUTION)

    if _dangerous_reason(seg):
        return RiskLevel.DANGEROUS

    exe = tokens[0].lstrip("\\")
    if exe in safe_commands:
        return RiskLevel.SAFE
    if len(tokens) >= 2 and (exe, tokens[1]) in _BUILTIN_SAFE_PREFIXES:
        return RiskLevel.SAFE

    return RiskLevel.CAUTION


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

    # Split on compound separators FIRST (grouping- and quote-aware, so a
    # separator inside `(...)`, `{...}`, or quotes is never split). Only then
    # does each segment strip its own grouping and recurse. A leading group
    # segment (`(ls)`) followed by a separator no longer forces a whole-command
    # bailout: the later segment (which may be BLOCKED) is classified on its own.
    segments = _split_segments(command)
    if not segments:
        return RiskLevel.SAFE

    worst = RiskLevel.SAFE
    for segment in segments:
        worst = max(worst, _classify_segment(segment, safe_commands, blocked_patterns, depth + 1))
    return worst


def classify_command(command: str, config: dict[str, Any]) -> RiskLevel:
    """Classify a shell command into a RiskLevel.

    Compound commands (`;`, `&&`, `||`, `|`) are split and every segment is
    classified; the highest (strictest) risk wins. Wrapper prefixes, subshell
    groups, and command substitutions are recursively descended so the wrapped
    /grouped/substituted command carries its own risk (a wrapper can never let
    a blocked exe escape). A shlex parse error on the whole command is
    DANGEROUS. Config extends the built-in safe/blocked lists.
    """
    safety = config.get("safety", {}) if config else {}
    blocked_patterns = list(safety.get("blocked_patterns") or [])
    safe_commands = _BUILTIN_SAFE_COMMANDS | frozenset(safety.get("safe_commands") or [])

    if not command or not command.strip():
        return RiskLevel.SAFE

    # A parse failure on the whole command means we cannot reason about it.
    try:
        shlex.split(command)
    except ValueError:
        return RiskLevel.DANGEROUS

    worst = _classify(command, safe_commands, blocked_patterns, 0)

    # Belt-and-suspenders: block check against the whole command, so patterns
    # spanning a redirect (`echo hi > /dev/sda`) are caught even if segment
    # splitting rearranged them.
    if _blocked_reason(command, blocked_patterns) is not None:
        return RiskLevel.BLOCKED

    return worst


def _block_reason_recursive(command: str, blocked_patterns: list[str], depth: int) -> str | None:
    """Find the specific BLOCKED reason, descending wrappers/groups/subs so a
    wrapped/grouped/substituted blocked command reports its real reason."""
    if depth > _MAX_DEPTH or not command or not command.strip():
        return None
    cmd = command.strip()
    inner_group = _strip_grouping(cmd)
    if inner_group is not None:
        return _block_reason_recursive(inner_group, blocked_patterns, depth + 1)
    for segment in _split_segments(cmd):
        reason = _blocked_reason(segment, blocked_patterns)
        if reason:
            return reason
        for inner in _extract_substitutions(segment):
            reason = _block_reason_recursive(inner, blocked_patterns, depth + 1)
            if reason:
                return reason
        try:
            tokens = _strip_env_prefix(shlex.split(segment))
        except ValueError:
            continue
        remainder = _strip_wrapper_prefix(tokens)
        if isinstance(remainder, list) and remainder:
            inner_cmd = " ".join(shlex.quote(t) for t in remainder)
            reason = _block_reason_recursive(inner_cmd, blocked_patterns, depth + 1)
            if reason:
                return reason
        priv_payload = _strip_priv_prefix(tokens)
        if isinstance(priv_payload, list) and priv_payload:
            inner_cmd = " ".join(shlex.quote(t) for t in priv_payload)
            reason = _block_reason_recursive(inner_cmd, blocked_patterns, depth + 1)
            if reason:
                return reason
    return None


def block_reason(command: str, config: dict[str, Any]) -> str:
    """Return the human-readable reason a command is BLOCKED (for tool result)."""
    safety = config.get("safety", {}) if config else {}
    blocked_patterns = list(safety.get("blocked_patterns") or [])
    reason = _block_reason_recursive(command, blocked_patterns, 0)
    if reason:
        return reason
    reason = _blocked_reason(command, blocked_patterns)
    return reason or "blocked command"
