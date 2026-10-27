"""Stuck detection for autonomous runs: when the agent keeps failing, look the problem up on the web.

A local model that hits an error it doesn't know tends to retry the same thing with small variations.
`StuckWatcher` watches every tool result of a turn and decides when that is happening:

  * the same error signature again (seen twice), or three failed tool calls in a row  -> web lookup
  * the identical call three times in a row                                           -> change-of-approach nudge

The lookup is done by the harness (not left to the model's goodwill): it builds a search query from the
error line, with everything private stripped first (paths, user and host names, emails, tokens, IPs,
long quoted strings - the query goes to a public search engine), runs `web_search`, and hands the
results back as a system message. At most `max_lookups` per turn and one per distinct error.
Switch off with `autonomy.web_lookup: false` or `AGENT_AUTO_LOOKUP=0`.
"""

from __future__ import annotations

import getpass
import os
import re
import socket
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_FAIL_STREAK, _DEFAULT_REPEAT_SAME_ERROR, _DEFAULT_IDENTICAL_CALLS = 3, 2, 3
FAIL_STREAK = 3          # consecutive failed calls that count as stuck
REPEAT_SAME_ERROR = 2    # the same error signature this often counts as stuck
IDENTICAL_CALLS = 3
MAX_QUERY_CHARS = 140


def configure(config: dict[str, Any] | None) -> None:
    """Optional `autonomy.fail_streak` / `repeat_same_error` / `identical_calls` (the phone profile lowers them: a tiny model should be
    stopped and nudged sooner). Unset keys keep the module defaults above; bad values are ignored. A new StuckWatcher picks them up."""
    global FAIL_STREAK, REPEAT_SAME_ERROR, IDENTICAL_CALLS
    block = (config or {}).get("autonomy") if isinstance(config, dict) else None
    block = block if isinstance(block, dict) else {}

    def pick(key: str, default: int) -> int:
        value = block.get(key)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else default

    FAIL_STREAK = pick("fail_streak", _DEFAULT_FAIL_STREAK)
    REPEAT_SAME_ERROR = pick("repeat_same_error", _DEFAULT_REPEAT_SAME_ERROR)
    IDENTICAL_CALLS = pick("identical_calls", _DEFAULT_IDENTICAL_CALLS)

_EXIT_CODE = re.compile(r"exit code (-?\d+)")
_ERROR_LINE = re.compile(
    r"(?i)(error|exception|traceback|failed|failure|fatal|not found|no such file|cannot|can't|unable to|"
    r"undefined|unresolved|denied|refused|invalid|unexpected|assertion|segmentation|panic|conflict)"
)
_TOOL_FAILURE_START = re.compile(r"(?i)^\s*(error|could not|cannot|failed|unable|no such|not found|invalid)\b")
_NOISE_LINE = re.compile(r"^\s*(\$ |\(in |\^+\s*$|[-=]{3,}|\s*$)")

_PATH = re.compile(r"(?:(?<=\s)|^|(?<=[\"'(=:]))(?:~|/)[\w.\-+@%~/]*[\w.\-+@%]/[\w.\-+@%/~]*")
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")
_TOKEN = re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{24,}\b|\b[0-9a-fA-F]{16,}\b")
_ADDR = re.compile(r"\b0x[0-9a-fA-F]+\b")
_LINENO = re.compile(r"(?i)\b(line|zeile|col|column)\s+\d+\b|:\d+(:\d+)?\b")
_QUOTED_LONG = re.compile(r"([\"'])[^\"']{40,}\1")

_CONTEXT_WORDS = (
    (re.compile(r"\b(python3?|pytest|pip3?|uv|poetry|conda)\b"), "python"),
    (re.compile(r"\b(npm|npx|node|yarn|pnpm|tsc|deno|bun)\b"), "node"),
    (re.compile(r"\b(cargo|rustc|rustup)\b"), "rust"),
    (re.compile(r"\b(nix|nixos-rebuild|nix-shell|home-manager)\b"), "nixos"),
    (re.compile(r"\b(go)\s+(build|run|test|mod|get)\b"), "golang"),
    (re.compile(r"\b(gcc|g\+\+|clang|make|cmake)\b"), "c++"),
    (re.compile(r"\bgit\b"), "git"),
    (re.compile(r"\b(docker|podman)\b"), "docker"),
    (re.compile(r"\b(ffmpeg|ffprobe)\b"), "ffmpeg"),
    (re.compile(r"\b(systemctl|journalctl)\b"), "systemd"),
)


def sanitize(text: str) -> str:
    """Strip everything private from an error text before it is sent to a public search engine."""
    out = text
    for private in {getpass.getuser(), socket.gethostname(), os.path.basename(os.path.expanduser("~"))}:
        if private and len(private) > 2:
            out = re.sub(re.escape(private), " ", out, flags=re.IGNORECASE)
    for pattern, repl in ((_URL, " "), (_EMAIL, " "), (_QUOTED_LONG, " "), (_PATH, " "), (_IP, " "), (_TOKEN, " "),
                          (_ADDR, " "), (_LINENO, " ")):
        out = pattern.sub(repl, out)
    out = re.sub(r"[\"'`]", " ", out)
    return re.sub(r"\s+", " ", out).strip(" :-,;.")


def _meaningful_lines(result: str) -> list[str]:
    return [ln.strip() for ln in result.splitlines() if ln.strip() and not _NOISE_LINE.match(ln)]


def failure_signature(name: str, arguments: dict[str, Any], result: str) -> str | None:
    """The key error line of a failed tool call, or None if the call did not fail."""
    result = result or ""
    if name == "run_command":
        m = _EXIT_CODE.search(result[:400])
        if not m or int(m.group(1)) == 0:
            return None
        lines = _meaningful_lines(result)
        errors = [ln for ln in lines if _ERROR_LINE.search(ln)]
        picked = errors[-1] if errors else (lines[-1] if lines else f"exit code {m.group(1)}")
        return picked[:300]
    if _TOOL_FAILURE_START.match(result) or result.startswith("BLOCKED"):
        return result.strip().splitlines()[0][:300]
    return None


def error_query(signature: str, command: str = "") -> str:
    """Search query for an error line: sanitized, with the language/tool as context."""
    core = sanitize(signature)[:MAX_QUERY_CHARS]
    context = ""
    for pattern, word in _CONTEXT_WORDS:
        if pattern.search(command or ""):
            context = word
            break
    if context and context not in core.lower():
        core = f"{context} {core}"
    return core.strip()


def _norm(signature: str) -> str:
    return sanitize(signature).lower()[:160]


@dataclass
class Verdict:
    kind: str                  # "lookup" | "repeat"
    query: str = ""
    message: str = ""          # for "repeat": the nudge text
    count: int = 1


def num_of(v: Any) -> int:
    try:
        return max(int(v), 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class StuckWatcher:
    max_lookups: int = 3
    lookups: int = 0
    streak: int = 0
    last_call: tuple[str, str] | None = None
    identical: int = 0
    seen: dict[str, int] = field(default_factory=dict)
    looked_up: set[str] = field(default_factory=set)

    def state(self) -> dict[str, Any]:
        """The counters as plain JSON (a saved session keeps them: a restart is not a fresh allowance of lookups)."""
        return {"lookups": self.lookups, "streak": self.streak, "last_call": list(self.last_call) if self.last_call else None,
                "identical": self.identical, "seen": dict(self.seen), "looked_up": sorted(self.looked_up)}

    def load(self, d: Any) -> None:
        d = d if isinstance(d, dict) else {}

        self.lookups, self.streak, self.identical = num_of(d.get("lookups")), num_of(d.get("streak")), num_of(d.get("identical"))
        lc = d.get("last_call")
        self.last_call = (str(lc[0]), str(lc[1])) if isinstance(lc, (list, tuple)) and len(lc) == 2 else None
        seen = d.get("seen")
        self.seen = {str(k): num_of(v) for k, v in seen.items()} if isinstance(seen, dict) else {}
        looked = d.get("looked_up")
        self.looked_up = {str(x) for x in looked} if isinstance(looked, (list, tuple, set)) else set()

    def observe(self, name: str, arguments: dict[str, Any], result: str, denied: bool = False) -> Verdict | None:
        """Feed one finished tool call. Returns what to do about it, or None."""
        if denied:
            self.streak = 0
            return None
        call = (name, repr(sorted(arguments.items(), key=lambda kv: kv[0])))
        self.identical = self.identical + 1 if call == self.last_call else 1
        self.last_call = call
        signature = failure_signature(name, arguments, result)
        if signature is None:
            self.streak = 0
            self.identical, self.last_call = 0, None          # a success breaks any run of identical calls
            return None
        self.streak += 1
        key = _norm(signature)
        self.seen[key] = self.seen.get(key, 0) + 1

        stuck = self.seen[key] >= REPEAT_SAME_ERROR or self.streak >= FAIL_STREAK
        if stuck and key not in self.looked_up and self.lookups < self.max_lookups and name not in ("web_search", "web_fetch"):
            self.looked_up.add(key)
            self.lookups += 1
            query = error_query(signature, str(arguments.get("command", "")))
            if len(query) >= 8:
                return Verdict("lookup", query=query, count=self.seen[key])
        if self.identical >= IDENTICAL_CALLS:
            self.identical = 0
            return Verdict("repeat", message=(
                f"You have made the identical {name} call {IDENTICAL_CALLS} times and it keeps failing "
                "with the same result. Do not try it again. Read the error, question your assumptions, and "
                "change the approach (different command, different file, a smaller test), or look it up "
                "with web_search / web_fetch."
            ))
        return None


def lookup_message(query: str, count: int, results: str) -> str:
    return (
        f"AUTOMATIC WEB LOOKUP. Your last steps failed with the same problem ({count}x), so the search "
        f"'{query}' was run for you. The results below are live web data. Use what applies; do NOT repeat "
        "the failed approach. If they don't help, refine the query yourself with web_search or open a "
        "promising page with web_fetch.\n\n" + results
    )


def enabled(config: dict[str, Any]) -> bool:
    if os.environ.get("AGENT_AUTO_LOOKUP", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    return bool((config.get("autonomy") or {}).get("web_lookup", True))
