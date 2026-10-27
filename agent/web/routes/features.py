"""Optional GUI features: the screens and tabs that exist only when their route module is installed.

app.py collects the FEATURE dict of every route module that has one; the GUI asks here once at start and adds the
navigation entries / tabs it lists. Nothing is listed when no such module is present."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["features"])


@router.get("/features")
def features(request: Request) -> list[dict[str, Any]]:
    return list(getattr(request.app.state, "features", []))
