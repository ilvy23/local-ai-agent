"""Depth-scanning JSON extraction shared by distillation and tool-call fallback.

LLM output is rarely clean JSON: it may be fenced in a code block, prefixed
with chatty text, or contain an earlier bracket pair that isn't valid JSON.
`extract_json_value` scans for balanced `opener`/`closer` pairs left to right
and returns the first one that both parses and is JSON-decodable, skipping
over any that don't parse.
"""

from __future__ import annotations

import json
from typing import Any

# Valid JSON string escapes are \" \\ \/ \b \f \n \r \t and \uXXXX. Some
# models (e.g. mixtral) emit stray backslashes before other characters, most
# often "\_", which is not valid JSON and makes an otherwise well-formed
# object fail to parse. json.JSONDecodeError reports this specific defect
# with the message below and the offending backslash's position in `e.pos`.
_INVALID_ESCAPE_MSG = "Invalid \\escape"

# A model writing real multi-line content (a file's contents, in particular)
# into this fallback JSON-text format often emits a literal newline/tab
# instead of properly escaping it as \n/\t - JSON forbids raw control
# characters inside a string, and json reports that with this message.
# Unrepaired, this silently kills the whole tool call, which is very likely
# why models "learn" to keep tool-call file content to one line.
_INVALID_CONTROL_MSG = "Invalid control character"
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}

# Bound on repair rounds so pathological input can't loop indefinitely.
_MAX_JSON_REPAIRS = 20


def _escape_raw_control_chars_in_strings(candidate: str) -> str:
    """Escape literal newline/CR/tab characters found INSIDE a JSON string
    literal, leaving everything outside a string untouched.

    Tracks in-string state by toggling on unescaped double-quotes, treating
    a backslash as escaping exactly the next character (so `\"` inside a
    string doesn't get misread as the closing quote). This only rewrites
    control characters a real JSON parser would reject anyway - it can't
    turn invalid JSON into something that means something different, only
    into the well-formed version of what was clearly intended.
    """
    out = []
    in_string = False
    escaped = False
    for ch in candidate:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch in _CONTROL_ESCAPES:
                out.append(_CONTROL_ESCAPES[ch])
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)


def _repair_json(candidate: str) -> Any | None:
    """Try to parse `candidate`, repairing known LLM-shaped JSON defects
    (stray backslashes, raw control characters inside strings) as they show
    up, alternating repairs across a bounded number of rounds so a candidate
    with more than one defect still resolves. Returns None if it still
    doesn't parse, or a defect neither repair handles occurs.
    """
    for _ in range(_MAX_JSON_REPAIRS):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            if _INVALID_ESCAPE_MSG in e.msg:
                # e.pos is the index of the backslash itself; drop just that char.
                candidate = candidate[: e.pos] + candidate[e.pos + 1 :]
                continue
            if _INVALID_CONTROL_MSG in e.msg:
                repaired = _escape_raw_control_chars_in_strings(candidate)
                if repaired == candidate:
                    return None  # no progress possible; avoid an infinite loop
                candidate = repaired
                continue
            return None  # unrelated defect: don't attempt repair
    return None


def extract_json_value(text: str, opener: str, closer: str) -> Any | None:
    """Return the first balanced `opener...closer` span in `text` that parses
    as JSON, or None if none does.

    Depth-counting skips over `opener`/`closer` characters that fall INSIDE a
    JSON string literal (same in-string/escape tracking as
    _escape_raw_control_chars_in_strings above). Without this, a tool-call
    argument that's a shell command containing a literal brace - `find ...
    -exec ... {} \\;` is the classic case - miscounts depth, the whole
    extraction fails, and the model's raw JSON leaks into the chat as plain
    text instead of ever being recognized as a tool call.
    """
    start = text.find(opener)
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        repaired = _repair_json(candidate)
                        if repaired is None:
                            break  # not repairable: try the next `opener` after this start
                        return repaired
        start = text.find(opener, start + 1)
    return None
