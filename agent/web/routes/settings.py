"""Settings API: the same editable list `agent settings` shows (agent/settings.py), validated the same way."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agent import settings as S
from agent.config import DEFAULT_CONFIG
from agent.web.deps import get_config

router = APIRouter(prefix="/api/settings", tags=["settings"])

# a changed value is applied to the running server's config too; these only take effect after a restart
_RESTART = frozenset({"embed_model"})


# Settings that loosen the safety policy: the change is refused unless the caller confirms it (the GUI asks the user first).
# The built-in safe/blocked lists in agent.safety are never touched by config, so this only guards the user's own extensions.
_LOOSENING = {
    "safe_commands": lambda old, new: bool(set(new) - set(old)),
    "blocked_patterns": lambda old, new: bool(set(old) - set(new)),
    "safety_max_timeout_s": lambda old, new: int(new) > int(old),
}
_LOOSENING_TEXT = {
    "safe_commands": "These commands will run without asking you first.",
    "blocked_patterns": "The agent will stop refusing commands that match the removed patterns.",
    "safety_max_timeout_s": "A single shell command may run longer before it is killed.",
}


class SettingIn(BaseModel):
    value: Any
    confirm: bool = False


_WORDS = {"gpu": "GPU", "cpu": "CPU", "ram": "RAM", "gb": "(GB)", "mb": "(MB)", "s": "(seconds)", "k": "(count)", "wa": "WhatsApp", "whatsapp": "WhatsApp", "tier3": "Tier 3"}


def _label(name: str) -> str:
    words = [_WORDS.get(w, w) for w in name.split("_")]
    return " ".join(words)[0].upper() + " ".join(words)[1:]


def _item(name: str, meta: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": name, "category": meta["category"], "label": _label(name), "help": meta["help"],
        "kind": meta["kind"], "value": S._get(config, meta["path"]), "default": S._get(DEFAULT_CONFIG, meta["path"]),
        "choices": meta.get("choices"), "range": meta.get("range"), "restart_required": name in _RESTART,
        "loosens_safety": name in _LOOSENING, "note": _note(name),
    }


def _note(name: str) -> str | None:
    if name == "chat_context_window":
        from agent.llm import MAX_QWEN_CTX
        return (f"Every model is capped at {MAX_QWEN_CTX:,} tokens by default (models.max_ctx_ceiling), "
                "whatever is set here - add a models.overrides.<model>.num_ctx entry to raise it for one model.")
    return None


def _clean_list(value: Any) -> list[str]:
    """A list setting is a list of short single-line strings; blanks go, duplicates collapse, order stays."""
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise HTTPException(422, "Give a list of text entries.")
    out: list[str] = []
    for v in value:
        v = v.strip()
        if "\n" in v or len(v) > 200:
            raise HTTPException(422, "Each entry must be one line of at most 200 characters.")
        if v and v not in out:
            out.append(v)
    return out


def _guard(key: str, config: dict[str, Any], new: Any, confirm: bool) -> None:
    check = _LOOSENING.get(key)
    if check is None or confirm:
        return
    old = S._get(config, S.EDITABLE[key]["path"])
    if check(old if old is not None else ([] if isinstance(new, list) else new), new):
        raise HTTPException(409, "Confirmation needed. " + _LOOSENING_TEXT[key])


def _apply_live(config: dict[str, Any], dotted: str, value: Any) -> None:
    node = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _as_text(value: Any) -> Any:
    """The CLI hands every scalar over as text; JSON bodies carry real booleans and numbers."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return value


@router.get("")
def list_settings(config=Depends(get_config)) -> list[dict[str, Any]]:
    return [_item(name, meta, config) for name, meta in S.EDITABLE.items()]


@router.put("/{key}")
def put_setting(key: str, body: SettingIn, config=Depends(get_config)) -> dict[str, Any]:
    if key not in S.EDITABLE:
        raise HTTPException(404, f"No setting called {key!r}.")
    if S.EDITABLE[key]["kind"] == "list":
        value = _clean_list(body.value)
        _guard(key, config, value, body.confirm)
    else:
        value = _as_text(body.value)
        if not isinstance(value, str):
            raise HTTPException(422, "Give a single value.")
        checked, error = S.validate(key, value)
        if error:
            raise HTTPException(422, f"invalid value for '{key}': {error}")
        _guard(key, config, checked, body.confirm)
    try:
        dotted, coerced = S.set_value(key, value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    _apply_live(config, dotted, coerced)
    return _item(key, S.EDITABLE[key], config)


@router.post("/{key}/reset")
def reset_setting(key: str, confirm: bool = False, config=Depends(get_config)) -> dict[str, Any]:
    if key not in S.EDITABLE:
        raise HTTPException(404, f"No setting called {key!r}.")
    default = S._get(DEFAULT_CONFIG, S.EDITABLE[key]["path"])
    _guard(key, config, default, confirm)
    try:
        dotted, coerced = S.set_value(key, default if isinstance(default, list) else _as_text(default))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    _apply_live(config, dotted, coerced)
    return _item(key, S.EDITABLE[key], config)
