"""The embedding-model catalogue, its recommendation rules and `nightage embed list|use` (fake Ollama)."""

from __future__ import annotations

import copy

import pytest
from typer.testing import CliRunner

from agent import embedmodels as em
from agent.config import DEFAULT_CONFIG
from agent.memory.store import Store
from agent.memory.vectors import VectorIndex


def test_catalogue_is_complete_and_consistent():
    tags = [m.tag for m in em.CATALOGUE + em.OTHERS]
    assert len(tags) == len(set(tags))
    assert em.DEFAULT_TAG in tags and 4 <= len(em.curated()) <= 6
    for m in em.CATALOGUE:
        assert m.download_mb and m.ram_mb and m.dim and m.context, m.tag          # curated entries are fully measured
        assert m.languages in (em.ENGLISH, em.MULTILINGUAL) and m.label and m.tradeoff
    langs = {m.languages for m in em.curated()}
    assert langs == {em.ENGLISH, em.MULTILINGUAL}
    assert min(m.download_mb for m in em.curated()) < 100                          # a tiny one exists
    assert "English only" in em.describe("all-minilm").label
    assert "Multilingual" in em.describe("bge-m3").label and "German" in em.describe("bge-m3").label


def test_describe_variants_and_unknown():
    assert em.describe("bge-m3:latest").exact and em.describe("bge-m3:latest").tag == "bge-m3"
    v = em.describe("all-minilm:l12")
    assert v.languages == em.ENGLISH and not v.exact and v.tag == "all-minilm:l12"
    u = em.describe("some-new-embedder:1b")
    assert u.label == "custom embedding model" and u.languages == em.UNKNOWN and u.download_mb is None


@pytest.mark.parametrize("ram,hint,want", [
    (2000, "english", "all-minilm"),
    (8000, "en_US.UTF-8", "nomic-embed-text"),
    (3000, "de", "embeddinggemma:300m-qat-q4_0"),
    (8000, "fr_FR.UTF-8", "embeddinggemma:300m-qat-q4_0"),
    (32000, "de_DE.UTF-8", "bge-m3"),
    (32000, "english", "nomic-embed-text"),
    (8000, None, "embeddinggemma:300m-qat-q4_0"),
    (2000, None, "all-minilm"),
    (0, "multilingual", "all-minilm"),                  # unknown RAM: be careful ... multilingual below 4 GB -> small multilingual
])
def test_recommend(ram, hint, want):
    got = em.recommend(ram, 0, hint).tag
    if ram == 0 and hint == "multilingual":
        want = "embeddinggemma:300m-qat-q4_0"
    assert got == want


def test_warn_if_large():
    big = em.describe("bge-m3")
    assert "1186" in em.warn_if_large(big, 2000, 7000)
    assert em.warn_if_large(big, 30000, 32000) == ""
    assert em.warn_if_large(em.describe("all-minilm"), 1000, 4000) == ""
    assert em.warn_if_large(em.describe("mystery"), 100, 100) == ""


# ------------------------------------------------------------------ CLI

class FakeOllama:
    def __init__(self, installed, dim=384):
        self.installed, self.dim, self.embedded = dict.fromkeys(installed, 1), dim, []

    def installed_sizes(self):
        return self.installed

    def embed(self, texts, model, num_gpu=0):
        self.embedded.append(model)
        return [[0.1 + i * 1e-3] * self.dim for i, _ in enumerate(texts)]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    import agent.config as cfgmod
    import agent.embed_cli as ec
    import agent.main as main
    import agent.settings as settings

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data"]["db_path"] = str(tmp_path / "a.db")
    saved = {}
    fake = FakeOllama({"bge-m3:latest", "all-minilm:latest"})
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(main, "_open_llm", lambda: fake)
    monkeypatch.setattr(settings, "set_value", lambda name, v: (cfg["models"].__setitem__("embed", v), saved.update(embed=v)))
    monkeypatch.setattr(ec, "_machine_ram_mb", lambda: 8000)
    monkeypatch.setattr(ec, "_locale_hint", lambda: "en_US.UTF-8")
    monkeypatch.setattr("agent.foreground.active", lambda config: False)
    pulls = []
    monkeypatch.setattr(ec, "_pull", lambda tag: pulls.append(tag) or 0)
    return type("C", (), {"cfg": cfg, "fake": fake, "saved": saved, "pulls": pulls, "db": tmp_path / "a.db",
                          "run": staticmethod(lambda *a: CliRunner(env={"COLUMNS": "240"}).invoke(main.app, ["embed", *a]))})


def test_embed_list_shows_labels_installed_and_in_use(cli):
    r = cli.run("list")
    assert r.exit_code == 0, r.output
    out = " ".join(r.output.split())
    assert "English only - smallest, fastest" in out and "Multilingual" in out
    assert "bge-m3" in out and "in use" in out and "recommended" in out
    assert "installed" in out
    assert cli.run().exit_code == 0                      # bare `nightage embed` lists too


def test_use_pulls_after_confirmation_and_rebuilds_without_mixing_sizes(cli):
    with Store(cli.db) as store:
        sid = store.create_session()
        fid = store.add_fact("has a cat called Mira", sid)
        vec = VectorIndex(store)
        vec.add("fact", fid, "has a cat called Mira", [0.5] * 1024)      # an index built by the old 1024-dim model
    cli.fake.dim = 768
    r = cli.run("use", "nomic-embed-text", "--yes")
    assert r.exit_code == 0, r.output
    assert cli.pulls == ["nomic-embed-text"] and cli.saved["embed"] == "nomic-embed-text"
    with Store(cli.db) as store:
        vec = VectorIndex(store)
        assert vec.DIM == 768
        assert len(vec.search([0.3] * 768, k=3, kinds=["fact"])) == 1      # the old fact is findable at the new size
    assert "re-embedded" in r.output


def test_use_declined_download_changes_nothing(cli):
    import agent.main as main

    r = CliRunner().invoke(main.app, ["embed", "use", "nomic-embed-text"], input="n\n")
    assert r.exit_code == 1 and cli.pulls == [] and "embed" not in cli.saved


def test_use_refuses_while_a_chat_is_open(cli, monkeypatch):
    monkeypatch.setattr("agent.foreground.active", lambda config: True)
    r = cli.run("use", "all-minilm", "--yes")
    assert r.exit_code == 1 and "chat is open" in " ".join(r.output.split()) and "embed" not in cli.saved


def test_use_a_model_that_cannot_embed_changes_nothing(cli):
    def boom(texts, model, num_gpu=0):
        raise RuntimeError("no such model")
    cli.fake.embed = boom
    r = cli.run("use", "all-minilm", "--yes")
    assert r.exit_code == 1 and "Nothing was changed" in " ".join(r.output.split()) and "embed" not in cli.saved


def test_use_current_model_is_a_noop(cli):
    r = cli.run("use", "bge-m3")
    assert r.exit_code == 0 and "already" in r.output and cli.pulls == []
