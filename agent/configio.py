"""Write the user's config.yaml safely.

Several commands used to `yaml.dump(load_config())` back into config.yaml: that wrote the MERGED config - every shipped default,
`auto` model roles replaced by fixed names, a profile's overlay (`profile: phone` became permanent) - and dropped every comment.
Others re-dumped the file sorted. Everything now goes through here:

  read_user(path)              -> the user's own file as a dict ({} when missing or empty) - never the merged config
  set_keys(path, {dotted: v})  -> change exactly those keys: edited IN PLACE in the text when the file's layout allows it (comments,
                                  order and every other line stay), else the user's own dict is re-dumped with the change. The
                                  result is re-parsed and must equal "the old file + these changes", or nothing is written.
                                  The write is atomic (a crash leaves the old file or the new one, never half of it).
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import yaml

from agent import textfile

__all__ = ["read_user", "set_keys"]

_LOCK = threading.Lock()              # one writer at a time inside this process (web requests, the TUI, a worker thread)
_KEY_LINE = re.compile(r"^(?P<indent>[ ]*)(?P<key>[A-Za-z0-9_.\-]+):(?P<rest>.*)$")


def read_user(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(textfile.read(p)[0])
    return data if isinstance(data, dict) else {}


def _with(data: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    import copy

    out = copy.deepcopy(data)
    for dotted, value in values.items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = node[part] = {}
            node = child
        node[parts[-1]] = value
    return out


def _scalar(value: Any) -> str | None:
    """`value` as an inline YAML scalar / flow collection, or None when it has no safe one-line form (a multi-line string)."""
    if isinstance(value, str) and ("\n" in value or len(value) > 400):
        return None
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:/@+\-]*", value):
        try:
            if yaml.safe_load(f"k: {value}") == {"k": value}:
                return value                        # `chat: gpt-oss:20b`, not `chat: 'gpt-oss:20b'`
        except yaml.YAMLError:
            pass
    text = yaml.safe_dump([value], default_flow_style=True, allow_unicode=True, width=10**9).strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    return text[1:-1]


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """Index after the block that belongs to the key on line `start` (lines more indented than `indent`, blank lines, comments)."""
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and not ln.lstrip().startswith("#") and len(ln) - len(ln.lstrip(" ")) <= indent:
            break
        end += 1
    return end


def _edit_text(text: str, values: dict[str, Any]) -> str | None:
    """The file text with `values` applied in place, or None when the layout is not simple enough to edit safely."""
    lines = text.splitlines()
    for dotted, value in values.items():
        rendered = _scalar(value)
        if rendered is None:
            return None
        parts = dotted.split(".")
        lo, hi, indent = 0, len(lines), -1          # search window and the indent of its parent key
        for depth, part in enumerate(parts):
            found = None
            child_indent = None
            for i in range(lo, hi):
                m = _KEY_LINE.match(lines[i])
                if not m:
                    continue
                ind = len(m.group("indent"))
                if ind <= indent:
                    continue
                if child_indent is None:
                    child_indent = ind                  # the first key of this block sets its indentation
                if ind == child_indent and m.group("key") == part:
                    found = (i, ind, m.group("rest"))
                    break
            last = depth == len(parts) - 1
            if found is None:
                pad = " " * (child_indent if child_indent is not None else (indent + 2 if indent >= 0 else 0))
                if "\t" in pad:
                    return None
                new = [f"{pad}{part}: {rendered}"] if last else [f"{pad}{part}:"]
                if not last:
                    sub_pad = pad + "  "
                    rest_parts = parts[depth + 1:]
                    for k, sub in enumerate(rest_parts):
                        is_last = k == len(rest_parts) - 1
                        new.append(f"{sub_pad}{sub}: {rendered}" if is_last else f"{sub_pad}{sub}:")
                        sub_pad += "  "
                insert_at = lo if indent >= 0 else len(lines)
                lines[insert_at:insert_at] = new
                break
            i, ind, rest = found
            if last:
                comment = ""
                m = re.match(r"^\s*(?P<val>[^#]*?)(?P<c>\s+#.*)?$", rest)
                if m and m.group("c"):
                    comment = m.group("c")
                if _block_end(lines, i, ind) > i + 1 and not rest.strip():
                    return None                         # the key holds a nested block now: not a simple value to replace
                lines[i] = f"{' ' * ind}{part}: {rendered}{comment}"
                break
            if rest.strip() and not rest.strip().startswith("#"):
                return None                             # an inline mapping (`models: {chat: x}`): leave it to the re-dump
            lo, hi, indent = i + 1, _block_end(lines, i, ind), ind
    return "\n".join(lines) + "\n"


def set_keys(path: Path | str, values: dict[str, Any]) -> str:
    """Write `values` ({"models.chat": "x", "persona.name": "y"}) into the config file at `path`. Returns "text" (edited in
    place) or "dump" (re-dumped from the user's own dict). Raises ValueError (nothing written) when the file is not a mapping."""
    p = Path(path)
    with _LOCK:
        text = textfile.read(p)[0] if p.exists() else ""
        old = yaml.safe_load(text) if text.strip() else {}
        if old is None:
            old = {}
        if not isinstance(old, dict):
            raise ValueError(f"{p} is not a YAML mapping; fix or delete it first")
        want = _with(old, values)
        edited = _edit_text(text, values) if text.strip() else None
        if edited is not None:
            try:
                if (yaml.safe_load(edited) or {}) == want:
                    textfile.write_atomic(p, edited)
                    return "text"
            except yaml.YAMLError:
                pass
        p.parent.mkdir(parents=True, exist_ok=True)
        textfile.write_atomic(p, yaml.safe_dump(want, sort_keys=False, allow_unicode=True))
        return "dump"
