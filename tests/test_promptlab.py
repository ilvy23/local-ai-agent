"""Prompt studio: parsing + plumbing (fake LLM, no Ollama)."""

from __future__ import annotations

from agent import promptlab


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, model, num_ctx=None, num_gpu=None,
             temperature=None, extra_options=None):
        yield self.reply


CFG = {"models": {"chat": "x"}}


def test_questions_parsed():
    llm = FakeLLM("Who is it for?\nWhat features?\nWhich platform?")
    qs = promptlab.clarifying_questions(llm, CFG, "app")
    assert qs == ["Who is it for?", "What features?", "Which platform?"]


def test_questions_strip_numbering_and_drop_non_questions():
    llm = FakeLLM("1. Who is it for?\n- none\nsome statement without a mark\nWhich platform?")
    qs = promptlab.clarifying_questions(llm, CFG, "app")
    assert qs == ["Who is it for?", "Which platform?"]


def test_questions_capped_at_four():
    llm = FakeLLM("\n".join(f"Q{i}?" for i in range(8)))
    assert len(promptlab.clarifying_questions(llm, CFG, "app")) == 4


def test_improve_returns_text():
    assert promptlab.improve(FakeLLM("A sharp prompt."), CFG, "rough") == "A sharp prompt."


def test_plan_returns_structured_text():
    assert promptlab.plan(FakeLLM("Role: dev\nObjective: x"), CFG, "rough").startswith("Role:")


def test_anonymize_strips_and_uses_selected_strength():
    llm = FakeLLM("build a tool for a small team, test locally")
    out = promptlab.anonymize(llm, CFG, "my family doordash cam so we're happy", "poison")
    assert "family" not in out.lower() and "team" in out.lower()


def test_anonymize_unknown_strength_falls_back():
    # an unknown strength must not KeyError — falls back to heavy
    assert promptlab.anonymize(FakeLLM("ok"), CFG, "x", "bogus") == "ok"


def test_refine_applies_feedback():
    assert promptlab.refine(FakeLLM("revised"), CFG, "old", "make it shorter") == "revised"


def test_max_runs_five_passes_and_traces():
    final, tr = promptlab.anonymize_max(FakeLLM("rebuilt task"), CFG, "personal thing", trace=True)
    assert final == "rebuilt task"
    assert set(tr) == {"core", "keep", "strip", "draft"}


def test_max_via_strength_dispatch():
    assert promptlab.anonymize(FakeLLM("x"), CFG, "y", "max") == "x"


def test_sessions_save_load_and_update(tmp_path, monkeypatch):
    monkeypatch.setattr(promptlab, "_SESS_PATH", tmp_path / "s.json")
    sid = promptlab._save_session("improve", [{"role": "user", "content": "make me a website please"}])
    got = promptlab._load_sessions()
    assert len(got) == 1 and got[0]["id"] == sid and got[0]["mode"] == "improve"
    assert got[0]["title"].startswith("make me a website")
    # same id → update in place, not a new row
    promptlab._save_session("improve",
                            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], sid)
    got = promptlab._load_sessions()
    assert len(got) == 1 and got[0]["messages"][-1]["content"] == "y"


def test_load_sessions_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(promptlab, "_SESS_PATH", tmp_path / "nope.json")
    assert promptlab._load_sessions() == []
