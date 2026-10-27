"""Check and repair tool-call arguments BEFORE a tool runs.

Small local models get the call shape wrong in predictable ways: a number as a string, a missing required
argument, an argument the tool doesn't have. Without this the call reached the handler (after an approval
prompt) and died with a raw Python TypeError. Here it is repaired when unambiguous ("5" -> 5, "true" -> True,
a JSON string for an array) and otherwise refused with a precise message the model can act on - no prompt,
no wasted step.
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Any

_TRUE, _FALSE = {"true", "yes", "1"}, {"false", "no", "0"}


def _type_name(value: Any) -> str:
    return {bool: "boolean", int: "integer", float: "number", str: "string", list: "array", dict: "object"}.get(
        type(value), type(value).__name__)


def _coerce(value: Any, wanted: str) -> tuple[Any, bool]:
    """(value, ok) for the JSON-schema type `wanted`."""
    if wanted == "string":
        if isinstance(value, str):
            return value, True
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), True
        return value, False
    if wanted == "integer":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int):
            return value, True
        if isinstance(value, float) and value.is_integer():
            return int(value), True
        if isinstance(value, str):
            try:
                return int(value.strip()), True
            except ValueError:
                return value, False
        return value, False
    if wanted == "number":
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            return value, True
        if isinstance(value, str):
            try:
                return float(value.strip()), True
            except ValueError:
                return value, False
        return value, False
    if wanted == "boolean":
        if isinstance(value, bool):
            return value, True
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE, True
        return value, False
    if wanted in ("array", "object"):
        target = list if wanted == "array" else dict
        if isinstance(value, target):
            return value, True
        if isinstance(value, str):
            parsed = _parse_structured(value, target)
            return (parsed, True) if parsed is not None else (value, False)
        return value, False
    return value, True


def _parse_structured(text: str, target: type) -> Any | None:
    """A list/dict that arrived as TEXT. Strict JSON first; then the repairs local models need: raw newlines or bad escapes
    inside strings (code!), text around the value, and Python literals with single quotes."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, target):
            return parsed
    except ValueError:
        pass
    from agent.jsonx import extract_json_value

    opener, closer = ("[", "]") if target is list else ("{", "}")
    try:
        parsed = extract_json_value(text, opener, closer)
        if isinstance(parsed, target):
            return parsed
    except Exception:  # noqa: BLE001 - repair is best effort
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, target):
            return parsed
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    return None


def _example(spec: Any) -> str:
    """A tiny example value for a schema, so a refused call shows the model what to send."""
    if not isinstance(spec, dict):
        return '"..."'
    kind = spec.get("type")
    if kind == "array":
        return "[" + _example(spec.get("items")) + "]"
    if kind == "object":
        props = spec.get("properties") or {}
        return "{" + ", ".join(f'"{k}": {_example(v)}' for k, v in list(props.items())[:4]) + "}"
    return {"integer": "1", "number": "1.0", "boolean": "true"}.get(kind, '"..."')


def _signature(props: dict[str, Any], required: list[str]) -> str:
    parts = []
    for name, spec in props.items():
        kind = spec.get("type", "any") if isinstance(spec, dict) else "any"
        parts.append(f"{name} ({kind}{', required' if name in required else ''})")
    return ", ".join(parts)


def accepts_any_keyword(handler: Any) -> bool:
    try:
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        return True


def validate(tool: Any, arguments: Any) -> tuple[dict[str, Any], str | None]:
    """(arguments to use, error message or None). Never raises."""
    if not isinstance(arguments, dict):
        return {}, f"Error: invalid arguments for tool '{tool.name}': expected an object of named arguments."
    schema = getattr(tool, "parameters", None) or {}
    props = schema.get("properties")
    if schema.get("type") != "object" or not isinstance(props, dict):
        return arguments, None
    required = [r for r in schema.get("required", []) if isinstance(r, str)]
    fixed: dict[str, Any] = {}
    problems: list[str] = []
    lenient = accepts_any_keyword(tool.handler)
    for key, value in arguments.items():
        spec = props.get(key)
        if spec is None:
            if lenient:
                fixed[key] = value
            else:
                problems.append(f"unknown argument '{key}'")
            continue
        wanted = spec.get("type") if isinstance(spec, dict) else None
        if wanted:
            value, ok = _coerce(value, wanted)
            if not ok:
                problems.append(f"argument '{key}' must be {wanted}, got {_type_name(value)}")
                continue
        enum = spec.get("enum") if isinstance(spec, dict) else None
        if enum and value not in enum:
            problems.append(f"argument '{key}' must be one of {enum}, got {value!r}")
            continue
        fixed[key] = value
    missing = [r for r in required if r not in fixed and not any(p.startswith(f"argument '{r}'") for p in problems)]
    if missing:
        problems.insert(0, "missing required argument" + ("s " if len(missing) > 1 else " ") + ", ".join(missing))
    if problems:
        bad = [k for k in props if any(p.startswith(f"argument '{k}'") for p in problems)]
        hint = "".join(f" Example: \"{k}\": {_example(props[k])} (a real JSON {props[k].get('type')}, not a quoted string)."
                       for k in bad if isinstance(props[k], dict) and props[k].get("type") in ("array", "object"))
        return fixed, (f"Error: invalid arguments for tool '{tool.name}': {'; '.join(problems)}. "
                       f"Arguments: {_signature(props, required)}.{hint} Call it again with correct arguments.")
    return fixed, None
