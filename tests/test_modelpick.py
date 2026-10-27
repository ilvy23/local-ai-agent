"""agent/modelpick.py: every model role gets a model that is really installed; nothing needs config editing. Fake /api/tags only."""
from __future__ import annotations

import httpx
import pytest
import yaml

from agent import config as cfgmod
from agent import modelpick as mp

GB = 1024 ** 3
MB = 1024 ** 2
URL = "http://fake:11434"
CHAT = {"qwen3.5:0.8b": 1000 * MB}
EMB = {"all-minilm:latest": 46 * MB}


@pytest.fixture(autouse=True)
def _fresh():
    mp.forget_cache()
    yield
    mp.forget_cache()


def roles(res):
    return {r: p.model for r, p in res.roles.items()}


def test_defaults_resolve_to_the_only_chat_model_and_the_only_embedder():
    res = mp.resolve(cfgmod.DEFAULT_CONFIG["models"], set(), {**CHAT, **EMB}, ram_mb=4000)
    got = roles(res)
    assert all(got[r] == "qwen3.5:0.8b" for r in mp.CHAT_ROLES) and got["embed"] == "all-minilm:latest"
    assert not res.ask and len(res.notices) == 1 and "only models installed" in res.notices[0] and res.notices[0].count("\n") == 0


def test_embedding_models_are_never_chosen_for_chat():
    res = mp.resolve(cfgmod.DEFAULT_CONFIG["models"], set(), {**EMB, "bge-m3:latest": GB, "qwen3.5:4b": 3 * GB}, ram_mb=16000)
    assert roles(res)["chat"] == "qwen3.5:4b"
    chat, emb = mp.split({**EMB, "qwen3.5:4b": GB, "nomic-embed-text:latest": 1})
    assert chat == ["qwen3.5:4b"] and len(emb) == 2


def test_several_models_pick_best_that_fits_ram():
    inst = {"qwen3.5:0.8b": 1000 * MB, "qwen3.5:9b": 6600 * MB, "llama3:8b": 4700 * MB}
    assert mp.best_chat(inst, sorted(inst), ram_mb=7000) == "qwen3.5:0.8b"          # the 9b would eat more than 60 % of 7 GB
    assert mp.best_chat(inst, sorted(inst), ram_mb=32000) == "qwen3.5:9b"
    assert mp.best_chat(inst, [], ram_mb=1) == ""


def test_ollama_down_changes_nothing_and_never_blocks(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", boom)
    assert mp.installed(URL) is None
    res = mp.resolve(cfgmod.DEFAULT_CONFIG["models"], set(), None)
    assert res.reachable is False and roles(res)["chat"] == "qwen2.5:7b" and all(p.installed is None for p in res.roles.values())
    assert mp.known_missing(URL, "anything") is False


def test_tags_are_cached_for_thirty_seconds(monkeypatch):
    calls = []

    class R:
        def raise_for_status(self): ...
        def json(self): return {"models": [{"name": "a:1b", "size": 5}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: calls.append(1) or R())
    mp.installed(URL, now=100.0)
    mp.installed(URL, now=120.0)
    assert len(calls) == 1
    mp.installed(URL, now=131.0)
    assert len(calls) == 2


def test_explicit_missing_model_is_kept_and_reported_not_swapped():
    cfg = {**cfgmod.DEFAULT_CONFIG["models"], "chat": "llama3:8b"}
    res = mp.resolve(cfg, {"chat"}, {**CHAT, **EMB}, ram_mb=4000)
    assert res.roles["chat"].model == "llama3:8b" and res.roles["chat"].installed is False and res.roles["chat"].source == "config"
    assert ("chat", "llama3:8b") in res.ask
    assert res.roles["background"].model == "qwen3.5:0.8b"          # the other roles still work meanwhile


def test_embedder_is_not_switched_when_the_index_has_another_vector_size():
    res = mp.resolve(cfgmod.DEFAULT_CONFIG["models"], set(), {**CHAT, **EMB}, ram_mb=4000, index_dim=1024)
    assert res.roles["embed"].model == "bge-m3" and res.roles["embed"].installed is False
    assert any("nightage embed use" in n for n in res.notices)


def test_known_missing_only_with_a_fresh_answer(monkeypatch):
    mp._cache[URL] = (__import__("time").monotonic(), {"a:1b": 1})

    class R:
        def raise_for_status(self): ...
        def json(self): return {"models": [{"name": "a:1b", "size": 1}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())
    assert mp.known_missing(URL, "b:2b") is True and mp.known_missing(URL, "a:1b") is False
    assert mp.known_missing("http://other:1", "b:2b") is False      # nothing cached for it


def test_recommendation_by_ram_and_size_text():
    assert mp.recommend_chat(4000) == "qwen3.5:0.8b" and mp.recommend_chat(16000) == "qwen3.5:4b" and mp.recommend_chat(32000) == "qwen3.5:9b"
    assert mp.size_text("qwen3.5:0.8b").startswith("about 1.0") and mp.size_text("x:1b") == "size unknown"


def test_fresh_config_file_pins_no_model_and_roles_follow_what_is_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "index_dim", lambda c: None)
    path = tmp_path / "config.yaml"
    cfg = cfgmod.load_config(path, tmp_path / "data")
    written = yaml.safe_load(path.read_text())["models"]
    assert all(written[r] == "auto" for r in cfgmod.AUTO_ROLES)
    assert cfgmod.last_load()["user_models"] == []
    mp.resolve_config(cfg, URL, models={**CHAT, **EMB}, ram_mb=4000)
    assert cfg["models"]["chat"] == "qwen3.5:0.8b" and cfg["models"]["embed"] == "all-minilm:latest" and cfg["models"]["coding"] == "qwen3.5:0.8b"
    again = cfgmod.load_config(path, tmp_path / "data")                      # a second load of the same file: same answer, file untouched
    assert yaml.safe_load(path.read_text())["models"]["chat"] == "auto" and again["models"]["chat"] == cfgmod.DEFAULT_CONFIG["models"]["chat"]


def test_explicit_file_value_counts_as_chosen(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("models:\n  chat: llama3:8b\n  background: auto\n")
    cfg = cfgmod.load_config(path, tmp_path / "data")
    assert cfg["models"]["chat"] == "llama3:8b" and cfgmod.last_load()["user_models"] == ["chat"]


def test_set_models_edits_in_place_and_keeps_comments(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("# mine\nmodels:\n  chat: old:1b   # note\n  embed: x\nui:\n  a: 1\n")
    assert mp.set_models(p, {"chat": "new:2b", "coding": "new:2b"}) == "text"
    text = p.read_text()
    assert text.startswith("# mine") and "# note" in text and yaml.safe_load(text)["models"] == {"coding": "new:2b", "chat": "new:2b", "embed": "x"}
    q = tmp_path / "none.yaml"
    mp.use_everywhere(q, "m:1b", "e:1")
    m = yaml.safe_load(q.read_text())["models"]
    assert m["weekly"] == "m:1b" and m["embed"] == "e:1" and m["reasoning"] == "m:1b"


def test_settings_set_chat_model_persists_and_is_read_back(tmp_path, monkeypatch):
    from agent import settings

    path = tmp_path / "config.yaml"
    monkeypatch.setattr(settings, "DEFAULT_CONFIG_PATH", path)
    settings.set_value("chat_model", "qwen3.5:0.8b")
    cfg = cfgmod.load_config(path, tmp_path / "data")
    assert cfg["models"]["chat"] == "qwen3.5:0.8b" and "chat" in cfgmod.last_load()["user_models"]
    settings.set_value("coding_model", "qwen3.5:0.8b")                       # used to be ignored: models.coding overrides executor.model
    from agent.coding import config as cc

    assert cc.load(cfgmod.load_config(path, tmp_path / "data"))["executor"]["model"] == "qwen3.5:0.8b"


# ---- the start-up flow (ensure_models), all fakes -------------------------------------------------------------------------


class Machine:
    """A fake Ollama: a model list that pulls extend, scripted answers to questions."""

    def __init__(self, tmp_path, installed=None, answers=(), monkeypatch=None):
        self.models = dict(installed or {})
        self.answers = list(answers)
        self.asked: list[str] = []
        self.said: list[str] = []
        self.pulled: list[str] = []
        self.path = tmp_path / "config.yaml"
        monkeypatch.setattr(mp, "installed", lambda url, **k: dict(self.models) if self.models is not None else None)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.setattr(mp, "index_dim", lambda c: None)              # never look at a real memory database

    def confirm(self, q, default=True):
        self.asked.append(q)
        return self.answers.pop(0) if self.answers else default

    def pull(self, m):
        self.pulled.append(m)
        self.models[m] = 1000 * MB
        return True

    def run(self, config, **kw):
        return mp.ensure_models(config, URL, say=self.said.append, confirm=self.confirm, pull=self.pull, config_path=self.path, ram_mb=kw.pop("ram_mb", 4000),
                                models=None, **kw)


def fresh(tmp_path):
    return cfgmod.load_config(tmp_path / "config.yaml", tmp_path / "data")


def test_flow_one_chat_one_embedder_installed_is_silent_but_one_line(tmp_path, monkeypatch):
    m = Machine(tmp_path, {**CHAT, **EMB}, monkeypatch=monkeypatch)
    cfg = fresh(tmp_path)
    m.run(cfg)
    assert m.asked == [] and m.pulled == [] and len(m.said) == 1 and "qwen3.5:0.8b" in m.said[0]
    assert cfg["models"]["chat"] == "qwen3.5:0.8b"


def test_flow_explicit_missing_asks_once_and_persists_only_after_yes(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("models:\n  chat: llama3:8b\n")
    m = Machine(tmp_path, {**CHAT, **EMB}, answers=[False, False], monkeypatch=monkeypatch)
    cfg = fresh(tmp_path)
    with pytest.raises(mp.NoModelAvailable):
        m.run(cfg)
    assert m.asked[0] == "llama3:8b is not installed. Use qwen3.5:0.8b instead?"
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["models"]["chat"] == "llama3:8b"        # declined: nothing written
    m = Machine(tmp_path, {**CHAT, **EMB}, answers=[True], monkeypatch=monkeypatch)
    cfg = fresh(tmp_path)
    m.run(cfg)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["models"]["chat"] == "qwen3.5:0.8b" and cfg["models"]["chat"] == "qwen3.5:0.8b"
    assert len([q for q in m.asked if "not installed" in q]) == 1


def test_flow_no_keyboard_never_guesses_and_says_the_command(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("models:\n  chat: llama3:8b\n")
    m = Machine(tmp_path, {**CHAT}, monkeypatch=monkeypatch)
    m.confirm = lambda q, d=True: None
    with pytest.raises(mp.NoModelAvailable) as e:
        m.run(fresh(tmp_path))
    assert "nightage models use qwen3.5:0.8b" in e.value.message


def test_flow_empty_machine_end_to_end_offers_recommended_model_and_embedder(tmp_path, monkeypatch):
    m = Machine(tmp_path, {}, monkeypatch=monkeypatch)
    cfg = fresh(tmp_path)
    m.run(cfg)
    assert m.pulled[0] == "qwen3.5:0.8b" and len(m.pulled) == 2                       # chat model, then the memory model
    assert "No chat model is installed. Download qwen3.5:0.8b (about 1.0 GB)?" in m.asked
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())["models"]
    assert saved["chat"] == "qwen3.5:0.8b" and saved["background"] == "qwen3.5:0.8b" and saved["embed"] == m.pulled[1]
    assert cfg["models"]["chat"] == "qwen3.5:0.8b"


def test_flow_empty_machine_declined_stops_with_the_command(tmp_path, monkeypatch):
    m = Machine(tmp_path, {}, answers=[False], monkeypatch=monkeypatch)
    with pytest.raises(mp.NoModelAvailable) as e:
        m.run(fresh(tmp_path))
    assert "nightage pull qwen3.5:0.8b" in e.value.message and m.pulled == []


def test_flow_embedder_decline_is_remembered(tmp_path, monkeypatch):
    m = Machine(tmp_path, {**CHAT}, answers=[False], monkeypatch=monkeypatch)
    m.run(fresh(tmp_path))
    n = len(m.asked)
    m.run(fresh(tmp_path))
    assert len(m.asked) == n                                                     # not asked again


def test_flow_ollama_down_is_a_noop(tmp_path, monkeypatch):
    m = Machine(tmp_path, {}, monkeypatch=monkeypatch)
    m.models = None
    m.run(fresh(tmp_path))
    assert m.asked == [] and m.pulled == []


def test_friendly_handles_unread_streaming_404_and_hides_the_raw_text():
    from agent import friendly

    def handler(request):
        return httpx.Response(404, json={"error": "model 'x:1b' not found"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c, c.stream("POST", "http://h/api/chat") as r:
        exc = httpx.HTTPStatusError("Client error '404 Not Found' for url 'http://h/api/chat' https://developer.mozilla.org/x", request=r.request, response=r)
        line, hint = friendly.explain(exc)
    assert line == "The model x:1b is not installed." and "nightage pull x:1b" in hint
    for e, word in ((httpx.ConnectError("boom"), "Could not reach"), (httpx.ReadTimeout("t"), "too long")):
        assert word in friendly.explain(e)[0]
    assert "http" not in friendly.one_line(httpx.ConnectError("http://secret")).replace("nightage", "")
    assert friendly.once("k") is True and friendly.once("k") is False


def test_models_cli_show_use_and_refusals(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from agent import models_cli
    from agent.main import app

    path = tmp_path / "config.yaml"
    monkeypatch.setattr(cfgmod, "DEFAULT_CONFIG_PATH", path)
    monkeypatch.setattr(cfgmod, "DEFAULT_DATA_DIR", tmp_path / "data")
    orig = cfgmod.load_config
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: orig(path, tmp_path / "data"))
    monkeypatch.setattr(mp, "index_dim", lambda c: None)
    inst = {**CHAT, **EMB, "llama3:8b": 4000 * MB}
    monkeypatch.setattr(mp, "installed", lambda url, **k: dict(inst))
    run = CliRunner().invoke
    out = run(app, ["models"]).output
    assert "chat" in out and "qwen3.5:0.8b" in out and str(path) in out.replace("\n", "")
    assert run(app, ["models", "use", "nope:1b", "--yes"]).exit_code == 1
    assert run(app, ["models", "use", "all-minilm:latest", "--yes"]).exit_code == 1       # an embedder is not a chat model
    r = run(app, ["models", "use", "llama3:8b", "--yes"])
    assert r.exit_code == 0 and yaml.safe_load(path.read_text())["models"]["chat"] == "llama3:8b"
    assert yaml.safe_load(path.read_text())["models"]["weekly"] == "llama3:8b"
    assert run(app, ["models", "auto", "--yes"]).exit_code == 0 and yaml.safe_load(path.read_text())["models"]["chat"] == "auto"
    assert models_cli.models_app is not None
