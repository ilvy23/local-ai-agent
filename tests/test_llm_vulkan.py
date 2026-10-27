"""Routing of big (27B) models to the dual-GPU Vulkan Ollama server.

Everything here uses httpx.MockTransport and a patched subprocess: no real Ollama
is contacted and no server (or GPU) is touched.
"""

import json
from unittest.mock import patch

import httpx
import pytest

from agent import llm
from agent.llm import OllamaClient, uses_vulkan

DEFAULT = "http://localhost:11434"
VULKAN = "http://127.0.0.1:11436"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("AGENT_VULKAN_MODELS", "AGENT_VULKAN_OLLAMA_URL", "AGENT_VULKAN_START_CMD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_VULKAN_MODELS", r"qwen[^\s]*27b")   # these tests are about the narrow pattern; the default has its own tests below


def _chat_lines():
    return "\n".join(
        json.dumps(x)
        for x in ({"message": {"content": "hi"}, "done": False}, {"message": {"content": ""}, "done": True})
    )


class Fake:
    """Two fake Ollama servers behind one MockTransport; records every call."""

    def __init__(self, vulkan_up=True, default_resident=(), vulkan_resident=()):
        self.vulkan_up = vulkan_up
        self.resident = {11434: list(default_resident), 11436: list(vulkan_resident)}
        self.calls: list[tuple[int, str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((port, path, body))
        if port == 11436 and not self.vulkan_up:
            raise httpx.ConnectError("refused", request=request)
        if path == "/api/version":
            return httpx.Response(200, json={"version": "x"})
        if path == "/api/ps":
            models = [{"name": n, "size_vram": 1} for n in self.resident[port]]
            return httpx.Response(200, json={"models": models})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": f"m{port}"}]})
        if path == "/api/generate" and body.get("keep_alive") == 0:
            self.resident[port] = [n for n in self.resident[port] if n != body["model"]]
            return httpx.Response(200, json={})
        if path == "/api/chat":
            return httpx.Response(200, text=_chat_lines())
        if path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1]]})
        return httpx.Response(200, json={})

    def ports_for(self, path):
        return [p for p, pa, _ in self.calls if pa == path]


def _client(fake):
    return OllamaClient(base_url=DEFAULT, transport=httpx.MockTransport(fake))


# --------------------------------------------------------------------------- matching


@pytest.mark.parametrize("name", ["qwen3.8:27b", "qwen3:27b", "Qwen2.5-27B-instruct", "qwen3.5:27b-q4_K_M"])
def test_qwen_27b_names_use_vulkan(name):
    assert uses_vulkan(name)


@pytest.mark.parametrize("name", ["qwen3.5:9b", "gemma3:27b", "qwen3:4b", "qwen3-coder:30b", "deepseek-r1:7b", "gemma2:9b"])
def test_every_generating_model_uses_vulkan_by_default(name, monkeypatch):
    monkeypatch.delenv("AGENT_VULKAN_MODELS", raising=False)
    monkeypatch.setattr(llm, "vulkan_available", lambda: True)      # capability present; absence is tested below
    assert uses_vulkan(name)


@pytest.mark.parametrize("name", ["bge-m3", "nomic-embed-text", "qwen3-embedding:0.6b", ""])
def test_embedding_models_and_empty_names_stay_off_it(name, monkeypatch):
    monkeypatch.delenv("AGENT_VULKAN_MODELS", raising=False)
    monkeypatch.setattr(llm, "vulkan_available", lambda: True)
    assert not uses_vulkan(name)


def test_routing_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("AGENT_VULKAN_MODELS", "")
    assert not uses_vulkan("qwen3.8:27b")


def test_custom_pattern(monkeypatch):
    monkeypatch.setenv("AGENT_VULKAN_MODELS", "gemma3:27b")
    assert uses_vulkan("gemma3:27b") and not uses_vulkan("qwen3.8:27b")


def test_invalid_regex_does_not_raise(monkeypatch):
    monkeypatch.setenv("AGENT_VULKAN_MODELS", "(")
    assert not uses_vulkan("qwen3.8:27b")


# --------------------------------------------------------------------------- routing


def test_27b_chat_goes_to_vulkan_server():
    fake = Fake()
    list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    assert fake.ports_for("/api/chat") == [11436]


def test_small_model_chat_goes_to_default_server():
    fake = Fake()
    list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.5:9b"))
    assert fake.ports_for("/api/chat") == [11434]


def test_default_servers_vram_is_freed_before_27b_runs():
    fake = Fake(default_resident=["qwen3.5:9b"])
    list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    unloads = [(p, b["model"]) for p, pa, b in fake.calls if pa == "/api/generate"]
    assert (11434, "qwen3.5:9b") in unloads
    # and the unload happened before the chat request
    paths = [pa for _, pa, _ in fake.calls]
    assert paths.index("/api/generate") < paths.index("/api/chat")


def test_vulkan_models_are_freed_when_a_small_model_runs_afterwards():
    fake = Fake()
    c = _client(fake)
    list(c.chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    fake.resident[11436] = ["qwen3.8:27b"]
    list(c.chat([{"role": "user", "content": "x"}], "qwen3.5:9b"))
    assert fake.resident[11436] == []


def test_embedding_stays_on_the_default_server():
    fake = Fake()
    _client(fake).embed(["t"], "bge-m3")
    assert fake.ports_for("/api/embed") == [11434]


# --------------------------------------------------------------------------- start / fallback


def test_server_is_started_on_demand():
    fake = Fake(vulkan_up=False)

    def fake_run(cmd, **kw):
        fake.vulkan_up = True
        fake.started = cmd

    with patch.object(llm, "_vulkan_start_command", return_value=["ollama-vk", "list"]), \
         patch.object(llm.subprocess, "run", side_effect=fake_run) as run:
        list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    assert run.call_count == 1
    assert fake.ports_for("/api/chat") == [11436]


def test_falls_back_to_default_when_vulkan_cannot_start():
    fake = Fake(vulkan_up=False)
    with patch.object(llm, "_vulkan_start_command", return_value=None):
        list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    assert fake.ports_for("/api/chat") == [11434]


def test_start_failure_falls_back_and_does_not_raise():
    fake = Fake(vulkan_up=False)
    with patch.object(llm, "_vulkan_start_command", return_value=["nope"]), \
         patch.object(llm.subprocess, "run", side_effect=OSError("no such file")):
        list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    assert fake.ports_for("/api/chat") == [11434]


def test_running_server_is_not_restarted():
    fake = Fake(vulkan_up=True)
    with patch.object(llm, "_vulkan_start_command", return_value=["ollama-vk", "list"]), \
         patch.object(llm.subprocess, "run") as run:
        c = _client(fake)
        list(c.chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
        list(c.chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    run.assert_not_called()


# --------------------------------------------------------------------------- queries


def test_loaded_models_covers_both_servers():
    fake = Fake(default_resident=["qwen3.5:9b"], vulkan_resident=["qwen3.8:27b"])
    assert set(_client(fake).loaded_models()) == {"qwen3.5:9b", "qwen3.8:27b"}


def test_loaded_models_ignores_a_stopped_vulkan_server_and_never_starts_it():
    fake = Fake(vulkan_up=False, default_resident=["qwen3.5:9b"])
    with patch.object(llm.subprocess, "run") as run:
        assert _client(fake).loaded_models() == ["qwen3.5:9b"]
    run.assert_not_called()


def test_unload_goes_to_the_right_server():
    fake = Fake(default_resident=["qwen3.5:9b"], vulkan_resident=["qwen3.8:27b"])
    c = _client(fake)
    assert c.unload("qwen3.8:27b") and c.unload("qwen3.5:9b")
    assert fake.resident == {11434: [], 11436: []}


def test_unload_all_frees_everything():
    fake = Fake(default_resident=["qwen3.5:9b"], vulkan_resident=["qwen3.8:27b"])
    assert set(_client(fake).unload_all()) == {"qwen3.5:9b", "qwen3.8:27b"}
    assert fake.resident == {11434: [], 11436: []}


def test_check_model_loaded_sees_the_vulkan_server():
    fake = Fake(vulkan_resident=["qwen3.8:27b"])
    assert _client(fake).check_model_loaded("qwen3.8:27b")


def test_speculative_options_only_for_vulkan_models(monkeypatch):
    from agent import llm

    monkeypatch.delenv("AGENT_DRAFT_TOKENS", raising=False)
    monkeypatch.setenv("AGENT_VULKAN_MODELS", r"qwen[^\s]*27b")
    assert llm.speculative_options("qwen3.8:27b") == {"draft_num_predict": 4}
    assert llm.speculative_options("qwen3:4b") == {}
    monkeypatch.setenv("AGENT_DRAFT_TOKENS", "3")
    assert llm.speculative_options("qwen3.8:27b") == {"draft_num_predict": 3}
    monkeypatch.setenv("AGENT_DRAFT_TOKENS", "-1")
    assert llm.speculative_options("qwen3.8:27b") == {}


def test_background_chat_is_capped_and_cpu_only_on_vulkan(monkeypatch):
    from agent import llm

    seen = {}

    class Resp:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_lines(self): return iter(['{"message": {"content": "ok"}, "done": true}'])

    class FakeClient:
        def stream(self, method, path, json=None, **kw):
            seen["payload"] = json
            return Resp()

    client = llm.OllamaClient.__new__(llm.OllamaClient)
    monkeypatch.setattr(llm.OllamaClient, "_client_for", lambda self, model: FakeClient())
    monkeypatch.setenv("AGENT_VULKAN_MODELS", ".*")
    monkeypatch.delenv("AGENT_BG_CPU", raising=False)
    try:
        "".join(client.chat([{"role": "user", "content": "x"}], "qwen3:4b", stream=False, background=True))
    except Exception:
        pass
    opts = seen["payload"]["options"]
    assert opts["num_gpu"] == 0 and opts["num_predict"] == llm.BACKGROUND_NUM_PREDICT
    monkeypatch.setenv("AGENT_BG_CPU", "0")
    try:
        "".join(client.chat([{"role": "user", "content": "x"}], "qwen3:4b", stream=False, background=True))
    except Exception:
        pass
    assert "num_gpu" not in seen["payload"]["options"]


def test_background_think_off_only_for_clean_models():
    from agent import llm

    assert llm.supports_think_off("qwen3.8:27b-text")
    assert llm.supports_think_off("qwen3.5:9b")
    assert not llm.supports_think_off("qwen3:4b")
    assert not llm.supports_think_off("gemma2:9b")


def test_background_call_reuses_the_chat_context(monkeypatch):
    from agent import llm

    llm._last_num_ctx.clear()
    llm._remember_ctx("qwen3.8:27b-text", 65536)
    assert llm._last_num_ctx["qwen3.8:27b-text"] == 65536
    llm._remember_ctx("qwen3.8:27b-text", None)  # no ctx given: keep what we know
    assert llm._last_num_ctx["qwen3.8:27b-text"] == 65536


def test_system_messages_are_folded_so_the_model_actually_sees_them():
    from agent import llm

    msgs = [
        {"role": "system", "content": "rules"},
        {"role": "system", "content": "repo map"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "{\"tool\": \"run\"}"},
        {"role": "tool", "content": "exit code 1"},
        {"role": "system", "content": "VERIFY FAILED: fix it"},
    ]
    out = llm._adapt_messages("qwen3.8:27b", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "user"]       # no system message after the head
    assert out[0]["content"] == "rules\n\nrepo map"                                           # leading ones merged into one
    assert out[-1]["content"] == llm.HARNESS_PREFIX + "VERIFY FAILED: fix it"
    assert msgs[-1]["role"] == "system" and len(msgs) == 6                                    # the caller's list is untouched
    assert llm._adapt_messages("qwen3.8:27b", out) == out                                     # idempotent: stable prompt prefix
    assert llm._adapt_messages("m", [{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- context cap for qwen


def test_context_is_capped_for_every_model_not_just_qwen_named_ones():
    # cap_ctx used to special-case the model name ("qwen" in it); now it's generic -
    # agent.ctxbudget.budget_for decides for every model the same way. With plenty of
    # (fake) VRAM supplied, the only limit left is the ceiling (MAX_QWEN_CTX, kept as the
    # default for back-compat).
    from agent import ctxbudget
    from agent.gpumode import Gpu
    from agent.llm import MAX_QWEN_CTX, cap_ctx

    assert MAX_QWEN_CTX == 95000
    plenty = ctxbudget.HardwareSummary(gpus=[Gpu("fake", "amd", 64000, 0)])
    assert cap_ctx("qwen3.8:27b-text", 131072, hw_summary=plenty) == 95000
    assert cap_ctx("Qwen3.5:9b", 98304, hw_summary=plenty) == 95000
    assert cap_ctx("qwen3.5:9b", 32768, hw_summary=plenty) == 32768
    assert cap_ctx("qwen3.5:9b", None) is None
    assert cap_ctx("deepseek-r1:7b", 131072, hw_summary=plenty) == 95000


def test_explicit_model_override_wins_over_the_ceiling():
    from agent.llm import cap_ctx

    config = {"models": {"overrides": {"deepseek-r1:7b": {"num_ctx": 150000}}}}
    assert cap_ctx("deepseek-r1:7b", 131072, config=config) == 150000


def test_the_request_sent_to_ollama_carries_the_capped_context(monkeypatch):
    import json

    monkeypatch.setenv("AGENT_VULKAN_MODELS", "")          # routing is not the point here

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}, "done": True})

    client = OllamaClient(transport=httpx.MockTransport(handler))
    client.chat_with_tools([{"role": "user", "content": "x"}], "qwen3.8:27b-text", [], num_ctx=131072)
    list(client.chat([{"role": "user", "content": "x"}], "qwen3.5:9b", stream=False, num_ctx=120000))
    assert [r["options"]["num_ctx"] for r in seen] == [95000, 95000]


# --------------------------------------------------------------------------- opt-in by capability


def _no_vulkan_env(monkeypatch):
    for var in ("AGENT_VULKAN_MODELS", "AGENT_VULKAN_OLLAMA_URL", "AGENT_VULKAN_START_CMD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_vulkan_probe", None)


def test_no_gpu_no_server_means_no_routing_and_no_warning(monkeypatch, caplog):
    from agent import gpumode

    _no_vulkan_env(monkeypatch)
    monkeypatch.setattr(gpumode, "gpu_mode_available", lambda gpus=None: False)
    monkeypatch.setattr(llm, "_vulkan_start_command", lambda: ["x"])     # even a start command is not enough without a GPU
    assert not llm.uses_vulkan("qwen3.5:4b") and not llm.vulkan_available()
    c = OllamaClient(base_url=DEFAULT, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    assert c.server_url("qwen3.5:4b") == DEFAULT
    assert not [r for r in caplog.records if "Vulkan" in r.getMessage()]


def test_gpu_but_nothing_to_route_to(monkeypatch):
    from agent import gpumode

    _no_vulkan_env(monkeypatch)
    monkeypatch.setattr(gpumode, "gpu_mode_available", lambda gpus=None: True)
    monkeypatch.setattr(llm, "_vulkan_start_command", lambda: None)
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("no")))
    assert not llm.uses_vulkan("qwen3.5:9b")


def test_reachable_server_with_a_gpu_is_routed(monkeypatch):
    from agent import gpumode

    _no_vulkan_env(monkeypatch)
    monkeypatch.setattr(gpumode, "gpu_mode_available", lambda gpus=None: True)
    monkeypatch.setattr(llm, "_vulkan_start_command", lambda: None)
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: httpx.Response(200))
    assert llm.uses_vulkan("qwen3.5:9b") and not llm.uses_vulkan("bge-m3")


def test_unit_or_command_with_a_gpu_is_routed(monkeypatch):
    from agent import gpumode

    _no_vulkan_env(monkeypatch)
    monkeypatch.setattr(gpumode, "gpu_mode_available", lambda gpus=None: True)
    monkeypatch.setattr(llm, "_vulkan_start_command", lambda: ["systemctl", "--user", "start", "ollama-vk.service"])
    assert llm.uses_vulkan("qwen3.5:27b")


def test_env_url_counts_as_configured_without_any_probe(monkeypatch):
    _no_vulkan_env(monkeypatch)
    monkeypatch.setenv("AGENT_VULKAN_OLLAMA_URL", "10.0.0.5:11436")
    assert llm.vulkan_available() and llm.uses_vulkan("qwen3.5:9b")
