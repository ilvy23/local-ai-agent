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

# Bound on repair iterations so pathological input (e.g. a long run of
# invalid escapes) can't loop indefinitely.
_MAX_ESCAPE_REPAIRS = 20


def _repair_invalid_escapes(candidate: str) -> str | None:
    """Iteratively drop backslashes that json reports as invalid escapes.

    Only ever touches the exact backslash json.JSONDecodeError points at, and
    only when the failure is specifically an invalid-escape error. Returns
    None if the candidate still doesn't parse as valid-escape JSON after
    `_MAX_ESCAPE_REPAIRS` attempts, or if a different kind of error occurs.
    """
    for _ in range(_MAX_ESCAPE_REPAIRS):
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError as e:
            if _INVALID_ESCAPE_MSG not in e.msg:
                return None
            # e.pos is the index of the backslash itself; drop just that char.
            candidate = candidate[: e.pos] + candidate[e.pos + 1 :]
    return None


def extract_json_value(text: str, opener: str, closer: str) -> Any | None:
    """Return the first balanced `opener...closer` span in `text` that parses
    as JSON, or None if none does.
    """
    start = text.find(opener)
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as e:
                        if _INVALID_ESCAPE_MSG not in e.msg:
                            break  # unrelated defect: don't attempt repair
                        repaired = _repair_invalid_escapes(candidate)
                        if repaired is None:
                            break
                        try:
                            return json.loads(repaired)
                        except json.JSONDecodeError:
                            break  # try the next occurrence of `opener` after this start
        start = text.find(opener, start + 1)
    return None
