"""Optional end-to-end test against a real local Ollama.

Excluded by default (`addopts = -m 'not integration'`). Run explicitly with:
    uv run pytest -m integration
Requires Ollama serving llama3.1:8b and nomic-embed-text.
"""

from __future__ import annotations

from io import StringIO

import httpx
import pytest
from rich.console import Console

from agent.config import DEFAULT_CONFIG
from agent.llm import OllamaClient
from agent.memory.recall import build_context
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex
from agent.tui import run_repl


class _ScriptedConsole(Console):
    def __init__(self, lines):
        super().__init__(file=StringIO(), force_terminal=False)
        self._lines = list(lines)

    def input(self, prompt="", **kwargs):
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


@pytest.mark.integration
def test_fact_recalled_in_new_session(tmp_path):
    try:
        OllamaClient().embed(["ping"], model="nomic-embed-text")
    except httpx.HTTPError:
        pytest.skip("Ollama not reachable")

    config = {**DEFAULT_CONFIG}
    config["models"] = {**config["models"], "chat": "llama3.1:8b"}
    client = OllamaClient()
    store = Store(tmp_path / "agent.db")

    run_repl(
        client,
        config,
        console=_ScriptedConsole(["my dog is called Rex", "he is a golden retriever", "/quit"]),
        store=store,
    )

    facts = " ".join(f["content"] for f in store.get_active_facts())
    assert "Rex" in facts

    vectors = VectorIndex(store)
    sid = store.create_session()
    messages = build_context(store, vectors, client, config, sid, "what's my dog's name?")
    assert "Rex" in messages[0]["content"]
    store.close()
