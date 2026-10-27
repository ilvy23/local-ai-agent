"""Turn a network / Ollama exception into ONE plain sentence plus a 'next:' hint. Nothing here ever shows a raw httpx message (those carry
a developer-docs URL and say nothing a user can act on). The full exception stays in the log (`/log`, logger output)."""

from __future__ import annotations

import re

import httpx

_MODEL_RE = re.compile(r"""model\s+['"]?([^'"\s]+)['"]?\s+not found|['"]([^'"]+)['"]\s+not found""", re.I)


def _detail(response: httpx.Response) -> str:
    """The server's own error text. A streamed response has not been read yet: read it now (small, an error body), never raise."""
    try:
        if not response.is_closed or not hasattr(response, "_content"):
            response.read()
    except Exception:  # noqa: BLE001 - already consumed / connection gone: fall through to what we have
        pass
    try:
        return str(response.json().get("error") or "")
    except Exception:  # noqa: BLE001
        try:
            return (response.text or "")[:200]
        except Exception:  # noqa: BLE001
            return ""


def missing_model(exc: BaseException) -> str:
    """The model name when `exc` is Ollama's 'model not found' (a 404), else ''."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        m = _MODEL_RE.search(_detail(exc.response))
        if m:
            return m.group(1) or m.group(2) or ""
    return ""


def explain(exc: BaseException, model: str | None = None) -> tuple[str, str]:
    """(one line, next hint). Plain text, no markup."""
    name = missing_model(exc)
    if name or (isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404 and model and "not found" in _detail(exc.response).lower()):
        name = name or model or "this model"
        return f"The model {name} is not installed.", f"nightage pull {name}   (or choose an installed one: nightage models)"
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        detail = " ".join(_detail(exc.response).split())[:120]
        if code == 404:
            return "Ollama does not know that request (404). Is it a very old version?", "nightage doctor"
        if code >= 500:
            return f"Ollama had a problem inside ({code}){': ' + detail if detail else ''}.", "nightage ollama status, then try again; the server log has details"
        return f"Ollama refused the request ({code}){': ' + detail if detail else ''}.", "nightage doctor"
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "Could not reach Ollama. Is it running?", "nightage ollama start"
    if isinstance(exc, httpx.TimeoutException):
        return "Ollama took too long to answer.", "wait and retry; a smaller model is faster (nightage models); nightage doctor"
    if isinstance(exc, httpx.HTTPError):
        return f"Network problem while talking to Ollama ({type(exc).__name__}).", "nightage doctor"
    return f"Something went wrong ({type(exc).__name__}).", "see /log, then retry"


def one_line(exc: BaseException, model: str | None = None) -> str:
    line, hint = explain(exc, model)
    return f"{line} next: {hint}"


# ------------------------------------------------------------------------------------------------------ logging and notices


def quiet_logging() -> str:
    """Interactive screens must never get a raw log line painted over them: without any handler Python prints WARNING and above to the
    terminal. Send them to a small rotating file instead ($XDG_STATE_HOME/nightage/nightage.log) unless the program (or a test) already
    configured logging. Idempotent. Returns the log file path ('' when it could not be created)."""
    import logging
    import logging.handlers

    root = logging.getLogger()
    if any(getattr(h, "_nightage", False) for h in root.handlers):
        return next(h.baseFilename for h in root.handlers if getattr(h, "_nightage", False) and hasattr(h, "baseFilename"))
    if root.handlers:
        return ""
    try:
        from agent import osport

        path = osport.state_home() / "nightage" / "nightage.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        h: logging.Handler = logging.handlers.RotatingFileHandler(path, maxBytes=200_000, backupCount=2, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        out = str(path)
    except OSError:
        h, out = logging.NullHandler(), ""
    h._nightage = True  # type: ignore[attr-defined]
    h.setLevel(logging.WARNING)
    root.addHandler(h)
    if root.level in (logging.NOTSET, logging.WARNING) or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    return out


_told: set[str] = set()


def once(key: str) -> bool:
    """True the first time `key` is asked in this process, False afterwards: for notices that must appear once per session."""
    if key in _told:
        return False
    _told.add(key)
    return True


def forget_notices() -> None:
    _told.clear()


def error_fields(exc: BaseException, model: str | None = None) -> dict:
    """Keywords for an `error` UI event. A network/Ollama error becomes one friendly line + a next hint (details go to the log file),
    and is marked as reported so the outer layers do not say it again; anything else keeps its own message."""
    if isinstance(exc, httpx.HTTPError):
        import logging

        logging.getLogger(__name__).warning("model request failed: %r", exc)
        line, hint = explain(exc, model)
        exc._nightage_reported = True  # type: ignore[attr-defined]
        return {"message": line, "next": hint, "code": "http"}
    return {"message": str(exc) or type(exc).__name__, "code": type(exc).__name__}
