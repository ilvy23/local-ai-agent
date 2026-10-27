from __future__ import annotations

import sys

import pytest


def pytest_collection_modifyitems(config, items):
    """Tests marked `posix_only` need bash / a pty / fcntl / /proc and are skipped on Windows (the CI matrix runs a subset there)."""
    if sys.platform != "win32":
        return
    skip = pytest.mark.skip(reason="needs a POSIX system")
    for item in items:
        if "posix_only" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_embedding_backoff(monkeypatch):
    """Don't make the suite wait out the embedding retry's backoff.

    `embed_with_retry` sleeps between attempts so a busy GPU has a moment to free
    up — real and wanted in production, pure dead time in tests (it took the suite
    from 2s to 96s). The retry logic itself still runs; only the waiting is gone.
    """
    monkeypatch.setattr("agent.memory.distill.time.sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def _isolated_gpu_choice(monkeypatch, tmp_path):
    """Tests never see the developer's real GPU choice and never start a real server.

    Without this, a CLI test runs the start-of-run GPU question non-interactively, which
    reuses the *real* last answer from ~/.local/state/agent/gpu_mode (e.g. "dual") for the
    rest of the process - and later tests would then try to start the real Vulkan Ollama
    through `ollama-vk` (a 90 s wait, and a real server on a real GPU)."""
    from agent import gpumode

    for var in ("AGENT_GPU_MODE", "AGENT_CHAT_CTX", "AGENT_SINGLE_KV", "AGENT_DUAL_KV",
                "AGENT_VULKAN_MODELS", "AGENT_VULKAN_OLLAMA_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_VULKAN_MODELS", r"qwen[^\s]*27b")      # tests keep the narrow routing; the real default (every model) is tested explicitly
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_VULKAN_START_CMD", "/bin/false")   # tests may never start a real server
    gpumode.reset()
    yield
    gpumode.reset()


@pytest.fixture(autouse=True)
def _isolated_code_policy(monkeypatch, tmp_path):
    """No test ever reads or writes the developer's real ~/.config/agent/code-policy.json.

    Found live: the suite had added ~12,500 `/tmp/pytest-...` repo entries to the real file (every test that grants a mode
    persists it). The policy path honours AGENT_CODE_POLICY, so every test gets its own file."""
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "code-policy.json"))


@pytest.fixture(autouse=True)
def _research_stays_offline(monkeypatch, tmp_path):
    """No test reaches the real network through the research tools or the developer's own trust rules.

    `connectors.lookup` runs on every web_search ("what is X" asks Wikipedia), the search-engine variables
    would route a test to a real SearXNG/Brave, and ~/.config/agent/source-trust.yaml would change the ratings.
    A test that wants a connector or an engine patches it again inside the test."""
    def offline(*args, **kwargs):
        raise RuntimeError("network is switched off in tests")

    monkeypatch.setattr("agent.research.connectors._get_json", offline)
    monkeypatch.setattr("agent.research.connectors._get_text", offline)
    for var in ("AGENT_SEARXNG_URL", "BRAVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("agent.research.trust.RULES_PATH", tmp_path / "no-source-trust.yaml")


@pytest.fixture(autouse=True)
def _no_real_plugins(monkeypatch, tmp_path):
    """Tests never load the developer's real plugins (the WhatsApp plugin reads private message data)."""
    monkeypatch.setenv("AGENT_PLUGIN_DIR", str(tmp_path / "no-plugins"))
