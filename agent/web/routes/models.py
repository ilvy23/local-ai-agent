"""The `/model` command's backend: list installed models, switch the
active one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agent.web.deps import get_client, get_config
from agent.web.schemas import ModelsOut, SelectModelIn

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=ModelsOut)
def list_models(client=Depends(get_client), config=Depends(get_config)) -> ModelsOut:
    try:
        available = client.list_models()
    except Exception:  # noqa: BLE001 - Ollama being down shouldn't 500 the page
        available = []
    return ModelsOut(
        current=config["models"]["chat"], available=available, loaded=client.loaded_models()
    )


@router.post("/select")
def select_model(
    body: SelectModelIn, client=Depends(get_client), config=Depends(get_config)
) -> dict[str, bool]:
    try:
        available = client.list_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Couldn't reach Ollama: {exc}") from exc
    if body.model not in available:
        raise HTTPException(400, f"'{body.model}' isn't an installed model.")

    from agent.settings import set_value

    previous_model = config["models"]["chat"]
    try:
        set_value("chat_model", body.model)
    except ValueError:
        pass  # switched for this process even if persisting the default failed
    config["models"]["chat"] = body.model

    # Switch the actual VRAM occupant right now, not lazily on the next
    # message - picking a smaller model specifically to free up room (e.g.
    # for image generation) should take effect the moment you pick it, not
    # whenever you next happen to type something.
    if previous_model != body.model:
        client.unload(previous_model)
        client.preload(body.model)
    return {"ok": True}


@router.post("/unload")
def unload_models(client=Depends(get_client)) -> dict[str, list[str]]:
    """Frees VRAM by evicting every resident model (chat + embedding) right
    now, on request - the manual counterpart to the automatic unload-on-
    switch above. The next message just reloads whatever's needed, same as
    any other lazy load - this only trades a bit of latency on the next
    request for freeing the GPU up in the meantime."""
    return {"unloaded": client.unload_all()}
