#!/usr/bin/env python3
"""Export training material from saved `agent code` sessions (read-only on the state dir).

    python scripts/export_trajectories.py [--state-dir ~/.local/state/agent] [--out data/trajectories]

Writes (JSONL, redacted):
  sft.jsonl    one record per successful session: the saved chat messages ("green" = see below)
  pairs.jsonl  one record per rejected/chosen pair taken around a loop nudge or a declined call

Only the saved sessions under <state-dir>/code-sessions/ are read (those are `agent code` sessions only;
the personal chat DB is never touched). Sessions store a *window* of the conversation: old tool outputs are
trimmed and early history is folded into a WORK LOG by the engine, so a record is the window, not the full run.

A session is "successful" when ALL hold:
  - meta state == "finished" and a model is recorded,
  - the last assistant message is plain text (no tool call),
  - there is green evidence: a harness line "All N items are confirmed" in a tool result, or a `run` result
    that is `exit code 0` with "passed" and no "failed"/"error",
  - no loop nudge / declined call inside the saved messages, and the event log (code-events.jsonl) has no
    `loop` / `loop_abort` event for that session id.

Pairs (for DPO/KTO): the trigger is a harness nudge system message ("You are repeating yourself", "You have sent
this command ... times", "N steps in a row without editing", ...) or a tool result starting with "NOT RUN"
(declined). rejected = the assistant call right before the trigger. chosen = the first assistant message after it
that (a) is a different call, (b) gets a non-error tool result, (c) is not followed by another trigger within 4
messages. Pairs without such a chosen message are dropped.

Format: messages keep the engine's wire format (assistant content is the JSON text {"tool":..,"arguments":..});
a parsed `tool_calls` list (OpenAI style) is added next to it so a trainer can re-render it with any chat template.
Redaction: agent/redact.py (secrets) + absolute home paths -> /home/user.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
try:  # reuse the project's secret redaction
    from agent.redact import redact as _redact
except Exception:  # pragma: no cover - keep the script usable without the package
    def _redact(text: str) -> str:
        return text

NUDGE_PREFIXES = (
    "You are repeating yourself",
    "You have sent this command",
    "You have run `",
    "Your last reply hit the output limit",
    "Only the first ",
)
NUDGE_RES = (
    re.compile(r"^\d+ steps in a row without editing"),
    re.compile(r"^\d+ read/search steps in a row"),
    re.compile(r"^You have spent \d+ steps reading"),
)
READ_TOOLS = {"read_file", "read_files", "search", "list_dir", "show_symbol", "repo_map", "outline", "where_used", "git_status"}
READ_NUDGES = ("steps in a row without editing", "read/search steps in a row", "steps reading")
ERROR_START = re.compile(r"^(Error|NOT RUN|edit_file failed|edit failed|\w+ failed)", re.I)
HOME_RE = re.compile(r"/home/[A-Za-z0-9._-]+")


def clean(text: str, home: str) -> str:
    user = Path(home).name if home else ""
    if home:
        text = text.replace(home, "/home/user")
    text = HOME_RE.sub("/home/user", text)
    if user:
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", "user", text)
    return _redact(text)


def is_nudge(msg: dict) -> bool:
    if msg.get("role") not in ("system", "user"):
        return False
    c = msg.get("content") or ""
    return c.startswith(NUDGE_PREFIXES) or any(r.match(c) for r in NUDGE_RES)


def is_declined(msg: dict) -> bool:
    return msg.get("role") == "tool" and (msg.get("content") or "").startswith("NOT RUN")


def parse_call(msg: dict) -> dict | None:
    if msg.get("role") != "assistant":
        return None
    c = (msg.get("content") or "").strip()
    if not c.startswith("{"):
        return None
    try:
        d = json.loads(c)
    except ValueError:
        return None
    if isinstance(d, dict) and isinstance(d.get("tool"), str):
        return d
    return None


def convert(msg: dict, home: str) -> dict:
    out = {"role": msg["role"], "content": clean(msg.get("content") or "", home)}
    call = parse_call(msg)
    if call is not None:
        args = call.get("arguments", {})
        out["tool_calls"] = [{"type": "function", "function": {
            "name": call["tool"], "arguments": json.loads(clean(json.dumps(args, ensure_ascii=False), home))
            if isinstance(args, dict) else args}}]
    return out


def same_call(a: dict | None, b: dict | None) -> bool:
    return a is not None and b is not None and a.get("tool") == b.get("tool") and a.get("arguments") == b.get("arguments")


def green_evidence(msgs: list[dict]) -> bool:
    for m in msgs:
        if m.get("role") != "tool":
            continue
        c = m.get("content") or ""
        if re.search(r"All \d+ items are confirmed|All items are confirmed", c):
            return True
        if c.startswith("exit code 0") and re.search(r"\b\d+ passed\b", c) and not re.search(r"\bfailed\b|\bERROR\b", c):
            return True
    return False


def successful(meta: dict, msgs: list[dict], bad_events: bool) -> tuple[bool, str]:
    if meta.get("state") != "finished":
        return False, f"state={meta.get('state')}"
    if not meta.get("model"):
        return False, "no model recorded"
    if bad_events:
        return False, "loop events in event log"
    if any(is_nudge(m) or is_declined(m) for m in msgs):
        return False, "nudge/declined in saved messages"
    last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    if last is None or parse_call(last) is not None:
        return False, "no plain-text final answer"
    if not green_evidence(msgs):
        return False, "no green evidence"
    return True, "ok"


def productive(msgs: list[dict], c: int) -> bool:
    """assistant message at index c is a real, non-erroring step."""
    if parse_call(msgs[c]) is None:
        return False
    res = next((m for m in msgs[c + 1:c + 3] if m.get("role") == "tool"), None)
    if res is None or ERROR_START.match((res.get("content") or "").strip()):
        return False
    j = c + 1
    for m in msgs[c + 1:c + 5]:
        if is_nudge(m) or is_declined(m):
            return False
        j += 1
    return True


def make_pairs(sid: str, msgs: list[dict], home: str) -> list[dict]:
    pairs = []
    used_rejected = set()
    for t, m in enumerate(msgs):
        if is_nudge(m):
            r = next((i for i in range(t - 1, -1, -1) if msgs[i].get("role") == "assistant"), None)
        elif is_declined(m):
            r = t - 1 if t > 0 and msgs[t - 1].get("role") == "assistant" else None
        else:
            continue
        if r is None or r in used_rejected:
            continue
        rej = parse_call(msgs[r])
        if rej is None:
            continue
        c = next((i for i in range(t + 1, len(msgs)) if msgs[i].get("role") == "assistant"), None)
        if c is None:
            continue
        ch = parse_call(msgs[c])
        if ch is None or same_call(ch, rej) or not productive(msgs, c):
            continue
        if is_nudge(m) and any(k in m["content"][:80] for k in READ_NUDGES) and ch["tool"] in READ_TOOLS:
            continue  # a "stop reading" nudge is only answered well by an edit/run/mark_done
        used_rejected.add(r)
        pairs.append({
            "id": f"{sid}:{t}",
            "source": sid,
            "trigger": "nudge" if is_nudge(m) else "declined",
            "prompt": [convert(x, home) for x in msgs[:r]],
            "rejected": convert(msgs[r], home),
            "between": [convert(x, home) for x in msgs[r + 1:t + 1]],
            "chosen": convert(msgs[c], home),
        })
    return pairs


def load_bad_event_sessions(events: Path) -> set[str]:
    bad: set[str] = set()
    if events.exists():
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") in ("loop", "loop_abort"):
                bad.add(str(e.get("session", "")).rsplit("-", 1)[-1])
    return bad


def export(state: Path, out: Path, home: str) -> dict:
    sess_dir = state / "code-sessions"
    bad = load_bad_event_sessions(state / "code-events.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    stats = {"sessions_seen": 0, "sft": 0, "pairs": 0, "pairs_nudge": 0, "pairs_declined": 0, "skipped": {}}
    sft, pairs = [], []
    for mp in sorted(sess_dir.glob("*/*.meta.json")):
        sid = mp.name.split(".")[0]
        body = mp.with_name(sid + ".json")
        if not body.exists():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            data = json.loads(body.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        msgs = [m for m in data.get("messages", []) if isinstance(m, dict) and "role" in m]
        stats["sessions_seen"] += 1
        if not meta.get("model"):
            stats["skipped"]["no model recorded"] = stats["skipped"].get("no model recorded", 0) + 1
            continue
        ok, why = successful(meta, msgs, sid in bad)
        if ok:
            sft.append({"id": sid, "source": sid, "task": clean(meta.get("title", ""), home),
                        "model": meta.get("model"), "messages": [convert(m, home) for m in msgs]})
        else:
            stats["skipped"][why] = stats["skipped"].get(why, 0) + 1
        pairs.extend(make_pairs(sid, msgs, home))
    with (out / "sft.jsonl").open("w", encoding="utf-8") as fh:
        for r in sft:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for r in pairs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["sft"] = len(sft)
    stats["pairs"] = len(pairs)
    stats["pairs_nudge"] = sum(1 for p in pairs if p["trigger"] == "nudge")
    stats["pairs_declined"] = sum(1 for p in pairs if p["trigger"] == "declined")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export SFT trajectories and rejected/chosen pairs from agent-code sessions.")
    ap.add_argument("--state-dir", type=Path, default=Path.home() / ".local" / "state" / "agent")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "trajectories")
    args = ap.parse_args(argv)
    stats = export(args.state_dir, args.out, str(Path.home()))
    print(f"sessions seen: {stats['sessions_seen']}")
    print(f"successful trajectories (sft.jsonl): {stats['sft']}")
    print(f"rejected/chosen pairs (pairs.jsonl): {stats['pairs']} (nudge {stats['pairs_nudge']}, declined {stats['pairs_declined']})")
    print(f"not exported as successful: {stats['skipped']}")
    print(f"output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
