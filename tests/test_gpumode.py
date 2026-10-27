"""The start-of-run "one GPU or both?" question and the context estimate behind it.

No Ollama, no GPU: HTTP is mocked, GPUs are plain dataclasses, prompts are injected.
"""

import io

import httpx
import pytest
from rich.console import Console

from agent import gpumode, llm
from agent.gpumode import DUAL, SINGLE, Gpu
from tests.test_llm_vulkan import (  # noqa: F401  (shared fake servers)
    Fake,
    _chat_lines,
    _client,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    gpumode.reset()
    for var in ("AGENT_GPU_MODE", "AGENT_CHAT_CTX", "AGENT_SINGLE_KV", "AGENT_DUAL_KV",
                "AGENT_VULKAN_MODELS", "AGENT_VULKAN_OLLAMA_URL", "AGENT_VULKAN_START_CMD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_VULKAN_MODELS", r"qwen[^\s]*27b")        # the gpumode tests are about the narrow name pattern
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    yield
    gpumode.reset()


# 27B-ish: 17 GB weights, 64 layers, 8 kv heads, head_dim 128 -> 64*8*256 elems per token
BIG = {"size_mb": 17000, "layers": 64, "kv_elems_per_token": 64 * 8 * 256, "ctx_max": 262144}
SMALL = {"size_mb": 5000, "layers": 36, "kv_elems_per_token": 36 * 8 * 256, "ctx_max": 40960}
RADEON = Gpu("RX 7800 XT", "amd", 16304, 1300)
GEFORCE = Gpu("RTX 4060 Ti", "nvidia", 8192, 20)


# ----------------------------------------------------------------------- estimate


def test_context_estimate_matches_the_kv_cache_arithmetic():
    est = gpumode.estimate_context(BIG, usable_mb=17000 + 8192, kv_type="f16")
    # 64*8*256 elems * 2 bytes = 262144 B = 0.25 MB per token -> 8192 MB / 0.25 = 32768
    assert est.fits and est.ctx == 32768 and not est.limited_by_model


def test_q8_kv_cache_gives_almost_twice_the_context():
    f16 = gpumode.estimate_context(BIG, 17000 + 8192, "f16").ctx
    q8 = gpumode.estimate_context(BIG, 17000 + 8192, "q8_0").ctx
    assert 1.8 < q8 / f16 < 1.95


def test_model_that_does_not_fit_reports_the_overflow():
    est = gpumode.estimate_context(BIG, usable_mb=14000, kv_type="f16")
    assert not est.fits and est.ctx == 0 and est.over_mb == 3000


def test_context_is_capped_by_the_models_own_maximum():
    est = gpumode.estimate_context(SMALL, usable_mb=60000, kv_type="q8_0")
    assert est.ctx == 40960 and est.limited_by_model


def test_context_is_rounded_down_to_a_multiple_of_1024():
    est = gpumode.estimate_context(BIG, 17000 + 3000, "f16")
    assert est.ctx % 1024 == 0 and est.ctx > 0


# ----------------------------------------------------------------------- options


def test_one_gpu_offers_only_a_single_option():
    opts = gpumode.build_options([RADEON])
    assert [o.mode for o in opts] == [SINGLE]


def test_two_gpus_offer_both_and_dual_has_more_room():
    single, dual = gpumode.build_options([RADEON, GEFORCE])
    assert (single.mode, dual.mode) == (SINGLE, DUAL)
    assert dual.usable_mb > single.usable_mb
    assert dual.kv_type == "q8_0" and single.kv_type == "f16"


def test_models_already_resident_count_as_free_vram():
    base = gpumode.build_options([RADEON, GEFORCE], resident_mb=0)[0].usable_mb
    freed = gpumode.build_options([Gpu("RX", "amd", 16304, 1300 + 9000), GEFORCE], resident_mb=9000)[0].usable_mb
    assert freed == pytest.approx(base, abs=1)


def test_kv_type_can_be_overridden(monkeypatch):
    monkeypatch.setenv("AGENT_SINGLE_KV", "q8_0")
    assert gpumode.build_options([RADEON, GEFORCE])[0].kv_type == "q8_0"


# ----------------------------------------------------------------------- model facts


def test_model_facts_are_read_from_show_and_tags():
    info = {
        "general.architecture": "qwen3",
        "qwen3.block_count": 40, "qwen3.attention.head_count": 32, "qwen3.attention.head_count_kv": 8,
        "qwen3.embedding_length": 5120, "qwen3.attention.key_length": 128, "qwen3.context_length": 40960,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": info})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m:1", "size": 9 * 1024**3}]})
        return httpx.Response(200, json={})

    facts = llm.OllamaClient(transport=httpx.MockTransport(handler)).model_facts("m:1")
    assert facts == {"size_mb": 9216, "layers": 40, "kv_elems_per_token": 40 * 8 * 256, "ctx_max": 40960, "approx": False}


def test_model_facts_handle_per_layer_kv_heads():
    info = {"general.architecture": "x", "x.block_count": 4, "x.attention.head_count": 8,
            "x.attention.head_count_kv": [8, 0, 8, 0], "x.attention.key_length": 64}

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": info})
        return httpx.Response(200, json={"models": [{"name": "h", "size": 1024**3}]})

    facts = llm.OllamaClient(transport=httpx.MockTransport(handler)).model_facts("h")
    assert facts["kv_elems_per_token"] == 2 * 8 * 128


def test_unknown_model_gives_none():
    def handler(request):
        return httpx.Response(404, json={})

    assert llm.OllamaClient(transport=httpx.MockTransport(handler)).model_facts("nope") is None


# ----------------------------------------------------------------------- the question


class FakeClient:
    def __init__(self):
        self.sizes = {"small:9b": 5000, "big:27b": 17000, "bge-m3": 1100}
        self.facts = {"small:9b": SMALL, "big:27b": BIG}

    def installed_sizes(self):
        return self.sizes

    def model_facts(self, model):
        return self.facts.get(model)

    def vram_in_use_mb(self):
        return 0


def _console():
    return Console(file=io.StringIO(), width=140, force_terminal=False)


def _choose(answer, ctx=None, config=None, **kw):
    seen = {}

    def ask(question, choices, default):
        seen["choices"], seen["default"] = choices, default
        return answer if answer is not None else default

    def ask_int(question, default):
        seen["ctx_default"] = default
        return ctx if ctx is not None else default

    console = _console()
    mode = gpumode.ensure_choice(FakeClient(), config or {"models": {"chat": "small:9b", "embed": "bge-m3"}},
                                 console, ask=ask, ask_int=ask_int, interactive=True,
                                 gpus=[RADEON, GEFORCE], **kw)
    return mode, seen, console.file.getvalue()


def test_question_shows_both_options_and_estimates_for_chat_and_biggest_model():
    mode, seen, out = _choose("2")
    assert mode == DUAL and seen["choices"] == ["1", "2"]
    assert "small:9b" in out and "big:27b" in out and "bge-m3" not in out
    assert "Tokens" in out


def test_default_is_dual_when_the_biggest_model_needs_both_cards():
    _, seen, _ = _choose(None)
    assert seen["default"] == "2"


def test_default_is_the_last_choice():
    _choose("1")
    gpumode.reset()
    _, seen, _ = _choose(None)
    assert seen["default"] == "1"


def test_chosen_context_is_applied_and_used_by_the_chat():
    from agent import tui

    _, seen, _ = _choose("2", ctx=65536)
    assert gpumode.chat_ctx_override() == 65536
    assert tui._num_ctx({"models": {"chat_num_ctx": 16384}}) == 65536
    assert seen["ctx_default"] >= 4096


def test_context_prompt_defaults_to_the_estimate_of_the_chat_model():
    _, seen, _ = _choose("2")
    est = gpumode.estimate_context(SMALL, gpumode.build_options([RADEON, GEFORCE])[1].usable_mb, "q8_0")
    assert seen["ctx_default"] == est.ctx


def test_question_is_asked_only_once():
    _choose("1")
    again = gpumode.ensure_choice(FakeClient(), {}, _console(), ask=lambda *a: pytest.fail("asked twice"),
                                  interactive=True, gpus=[RADEON, GEFORCE])
    assert again == SINGLE


def test_env_skips_the_question(monkeypatch):
    monkeypatch.setenv("AGENT_GPU_MODE", "dual")
    monkeypatch.setenv("AGENT_CHAT_CTX", "32768")
    mode = gpumode.ensure_choice(FakeClient(), {}, _console(), ask=lambda *a: pytest.fail("asked"),
                                 interactive=True, gpus=[RADEON, GEFORCE])
    assert mode == DUAL and gpumode.chat_ctx_override() == 32768


def test_one_gpu_means_no_question():
    mode = gpumode.ensure_choice(FakeClient(), {}, _console(), ask=lambda *a: pytest.fail("asked"),
                                 interactive=True, gpus=[RADEON])
    assert mode is None


def test_non_interactive_reuses_the_last_choice_without_asking():
    _choose("2")
    gpumode.reset()
    mode = gpumode.ensure_choice(FakeClient(), {}, _console(), ask=lambda *a: pytest.fail("asked"),
                                 interactive=False, gpus=[RADEON, GEFORCE])
    assert mode == DUAL and gpumode.chat_ctx_override() is None


def test_non_interactive_without_history_stays_undecided():
    assert gpumode.ensure_choice(FakeClient(), {}, _console(), interactive=False, gpus=[RADEON, GEFORCE]) is None


# ----------------------------------------------------------------------- routing follows the choice


def test_dual_mode_sends_even_small_models_to_the_vulkan_server():
    gpumode._state["mode"] = DUAL
    fake = Fake()
    list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.5:9b"))
    assert fake.ports_for("/api/chat") == [11436]


def test_single_mode_keeps_even_a_27b_on_the_default_server():
    gpumode._state["mode"] = SINGLE
    fake = Fake()
    list(_client(fake).chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    assert fake.ports_for("/api/chat") == [11434]


def test_embeddings_never_move_to_the_vulkan_server():
    gpumode._state["mode"] = DUAL
    fake = Fake()
    _client(fake).embed(["t"], "bge-m3")
    assert fake.ports_for("/api/embed") == [11434]


def test_no_choice_keeps_the_name_pattern_behaviour():
    fake = Fake()
    c = _client(fake)
    list(c.chat([{"role": "user", "content": "x"}], "qwen3.8:27b"))
    list(c.chat([{"role": "user", "content": "x"}], "qwen3.5:9b"))
    assert fake.ports_for("/api/chat") == [11436, 11434]


def test_a_huge_extra_model_that_fits_nowhere_is_not_listed():
    client = FakeClient()
    client.sizes["huge:70b"] = 40000
    client.facts["huge:70b"] = {"size_mb": 40000, "layers": 80, "kv_elems_per_token": 80 * 8 * 256, "ctx_max": 131072}
    options = gpumode.build_options([RADEON, GEFORCE])
    shown = [m for m, _ in gpumode.table_rows(client, {"models": {"chat": "small:9b"}}, options)]
    assert shown == ["small:9b"]


def test_gpu_names_carry_their_vram_size():
    assert gpumode.build_options([Gpu("RX 7800 XT 16 GB", "amd", 16304, 0), GEFORCE])[0].label == "RX 7800 XT 16 GB"


def test_table_lists_every_configured_model_including_small_ones():
    client = FakeClient()
    client.sizes["tiny:8b"] = 4800
    client.facts["tiny:8b"] = SMALL
    cfg = {"models": {"chat": "big:27b", "main": "small:9b", "coding": "tiny:8b", "embed": "bge-m3"}}
    options = gpumode.build_options([RADEON, GEFORCE])
    shown = [m for m, _ in gpumode.table_rows(client, cfg, options)]
    assert shown == ["big:27b", "small:9b", "tiny:8b"]


def test_chosen_context_is_clamped_to_each_models_own_maximum():
    client = FakeClient()
    cfg = {"models": {"chat": "big:27b", "main": "small:9b"}}
    gpumode.table_rows(client, cfg, gpumode.build_options([RADEON, GEFORCE]))   # fills the per-model facts
    gpumode._state["ctx"] = 200000
    assert gpumode.effective_ctx("small:9b", 16384) == 40960    # SMALL.ctx_max
    assert gpumode.effective_ctx("big:27b", 16384) == 200000
    assert gpumode.effective_ctx("unknown:1b", 16384) == 200000


def test_without_a_choice_the_configured_context_is_kept():
    assert gpumode.effective_ctx("small:9b", 16384) == 16384


def test_tui_num_ctx_uses_the_model_limit():
    from agent import tui

    gpumode.table_rows(FakeClient(), {"models": {"chat": "small:9b"}}, gpumode.build_options([RADEON, GEFORCE]))
    gpumode._state["ctx"] = 100000
    assert tui._num_ctx({"models": {"chat_num_ctx": 16384}}, "small:9b") == 40960


def test_hybrid_model_with_hidden_kv_heads_still_gets_a_conservative_estimate():
    # like qwen3.5: head_count_kv is hidden (null) in /api/show and only every 4th layer caches KV
    info = {"general.architecture": "qwen35", "qwen35.block_count": 32, "qwen35.attention.head_count": 16,
            "qwen35.attention.head_count_kv": None, "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256, "qwen35.full_attention_interval": 4,
            "qwen35.context_length": 262144}

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": info})
        return httpx.Response(200, json={"models": [{"name": "q", "size": 6 * 1024**3}]})

    facts = llm.OllamaClient(transport=httpx.MockTransport(handler)).model_facts("q")
    assert facts["approx"] is True
    assert facts["kv_elems_per_token"] == 8 * 16 * 512          # 32/4 = 8 attention layers, MHA assumed
    est = gpumode.estimate_context(facts, 6 * 1024 + 6000, "q8_0")
    assert est.fits and est.approx and 40000 < est.ctx <= 262144
    assert "vorsichtig" in gpumode._fmt_ctx(est)
