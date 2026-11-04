"""A model pulled only into the second (Vulkan) Ollama counts as installed (home screen, doctor, settings)."""

import httpx

from agent import modelpick


def test_models_of_the_vulkan_server_count_as_installed(monkeypatch):
    def get(url, timeout):
        names = ["gpt-oss:20b"] if ":11434" in url else ["qwen3.8:27b-text"]
        return httpx.Response(200, json={"models": [{"name": n, "size": 1} for n in names]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(modelpick.httpx, "get", get)
    monkeypatch.setenv("AGENT_VULKAN_OLLAMA_URL", "127.0.0.1:11436")
    from agent import gpumode
    monkeypatch.setattr(gpumode, "current_mode", lambda *a, **k: "dual")
    modelpick.forget_cache()
    got = modelpick.installed("http://127.0.0.1:11434")
    assert set(got) == {"gpt-oss:20b", "qwen3.8:27b-text"}
    assert set(modelpick.installed("http://127.0.0.1:11436")) == {"qwen3.8:27b-text"}
