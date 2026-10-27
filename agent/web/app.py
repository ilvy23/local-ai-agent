"""FastAPI app factory for the web chat frontend/backend.

Run with: uvicorn agent.web.app:app --host 0.0.0.0 --port 8420
(binding to all interfaces is fine here because the server itself is only
meant to be reachable via a private Tailscale network - see the deployment
plan, not because this app trusts its network.)
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from agent.web.auth import require_token
from agent.web.idle import touch_active
from agent.web import routes as _routes_pkg
from agent.web.routes import approvals, chat, files, host, images, memory, models, sessions, settings, system, workspace

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Agent Chat")

app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(approvals.router)
app.include_router(models.router)
app.include_router(files.router)
app.include_router(host.router)
app.include_router(images.router)
app.include_router(settings.router)
app.include_router(memory.router)
app.include_router(system.router)
app.include_router(workspace.router)

# Any further module in agent/web/routes that defines `router` is registered automatically (screens that need their own
# endpoints add a file there; nothing else has to change).
_EXPLICIT = {"approvals", "chat", "files", "host", "images", "memory", "models", "sessions", "settings", "system", "workspace"}
# A route module may also define FEATURE (a dict: key, label, and either nav {icon, after} or tab_of + module): the GUI
# lists those at /api/features to add a navigation entry or a tab. A module that cannot import (an optional piece that is
# not installed, as in the public export) is skipped, so its screens never appear and its endpoints answer 404.
app.state.features = []
for _mod in pkgutil.iter_modules(_routes_pkg.__path__):
    if _mod.name not in _EXPLICIT and not _mod.name.startswith("_"):
        try:
            _m = importlib.import_module(f"{_routes_pkg.__name__}.{_mod.name}")
        except ImportError:
            continue
        _router = getattr(_m, "router", None)
        if _router is not None:
            app.include_router(_router)
        if isinstance(getattr(_m, "FEATURE", None), dict):
            app.state.features.append(_m.FEATURE)


@app.get("/api/auth/check")
def auth_check() -> dict[str, bool]:
    """200 only when the request passed the token middleware: the GUI uses it to validate a typed-in token."""
    return {"ok": True}


_static_app = StaticFiles(directory=_STATIC_DIR, html=True)


app.middleware("http")(require_token)


@app.middleware("http")
async def _track_activity(request: Request, call_next):
    """Touches the idle-tracking file on every request - the idle-suspend
    script (scripts/agent-idle-suspend.sh) reads its mtime to decide whether
    anyone's actually using this before putting the machine to sleep. See
    agent/web/idle.py for why this is a plain file rather than a DB row."""
    touch_active()
    return await call_next(request)


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Without this, app.js/app.css/index.html get no Cache-Control header
    at all, so browsers fall back to heuristic caching - roughly 10% of the
    time since Last-Modified, per the HTTP spec's own suggested heuristic -
    and serve a stale copy from disk cache with NOT EVEN a conditional GET,
    let alone a real refetch. That means a frontend bug fix shipped here can
    silently keep failing to reach an already-open tab (very much including
    a "reload" on it) for a while after the fix lands, indistinguishable
    from the fix not having worked. `no-cache` (NOT no-store) still lets the
    browser cache the file - it just forces a conditional revalidation
    (If-None-Match/ETag, cheap 304 if unchanged) on every load, so this is
    guaranteed-fresh without giving up caching entirely. Media served via
    /api/files/{token} is untouched - those ARE meant to be cached hard,
    and already set their own headers via FileResponse."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/index.", "/js/", "/css/", "/manifest.", "/icon.")):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", _static_app, name="static")
