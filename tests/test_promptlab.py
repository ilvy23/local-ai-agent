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


# ── regex/heuristic PII net (model-free) ────────────────────────────────────

class EchoLLM:
    """Returns the exact prompt it was given, so tests can see whether
    redaction happened before the model was ever called."""
    def chat(self, messages, model, num_ctx=None, num_gpu=None,
             temperature=None, extra_options=None):
        yield messages[0]["content"]


def test_redact_obvious_email():
    out = promptlab._redact_obvious("contact bob@example.com for details")
    assert "bob@example.com" not in out and "[email]" in out


def test_redact_obvious_phone_but_not_dates_or_small_numbers():
    out = promptlab._redact_obvious("call +1 555-123-4567 about it")
    assert "555-123-4567" not in out and "[phone]" in out
    # a date-shaped or short number run must survive untouched
    assert promptlab._redact_obvious("due on 2024-01-15") == "due on 2024-01-15"
    assert promptlab._redact_obvious("it took 3-4 weeks") == "it took 3-4 weeks"


def test_redact_obvious_ipv4():
    out = promptlab._redact_obvious("the server is at 192.168.1.50 tonight")
    assert "192.168.1.50" not in out and "[ipv4]" in out


def test_redact_obvious_api_key_shaped_strings():
    out = promptlab._redact_obvious("token is ghp_abcdEFGH1234567890abcd, keep it safe")
    assert "ghp_abcdEFGH1234567890abcd" not in out and "[api_key]" in out


def test_redact_obvious_home_path_username():
    out = promptlab._redact_obvious("see /home/alice/Documents/notes.txt for context")
    assert "alice" not in out and "[unix_path]" in out


def test_redact_obvious_url_with_credentials():
    out = promptlab._redact_obvious("fetch https://user:secret@example.com/data")
    assert "secret" not in out and "[url_creds]" in out


def test_redact_obvious_leaves_normal_text_alone():
    text = "Build a small tool that watches a folder and renames files by date."
    assert promptlab._redact_obvious(text) == text


def test_leak_check_flags_residual_identifier():
    original = "email me at bob@example.com about the server at 10.0.0.5"
    leaked_result = "please reach bob@example.com about the network issue"
    leaks = promptlab.leak_check(original, leaked_result)
    assert any("bob@example.com" in l for l in leaks)


def test_leak_check_clean_when_nothing_survives():
    original = "email me at bob@example.com about the server at 10.0.0.5"
    clean_result = "reach out about the network issue"
    assert promptlab.leak_check(original, clean_result) == []


def test_leak_check_flags_home_dir_username():
    original = "logs are under /home/alice/data"
    leaked_result = "ask alice to check the data folder"
    leaks = promptlab.leak_check(original, leaked_result)
    assert any("alice" in l for l in leaks)


def test_anonymize_redacts_before_the_model_ever_sees_it():
    # EchoLLM hands back exactly what it received — proves redaction happens
    # pre-model, not just something the model happened to do.
    out = promptlab.anonymize(EchoLLM(), CFG,
                              "contact bob@example.com or +1 555-123-4567", "light")
    assert "bob@example.com" not in out
    assert "555-123-4567" not in out


def test_anonymize_max_redacts_before_the_model_ever_sees_it():
    final, trace = promptlab.anonymize_max(EchoLLM(), CFG,
                                           "ping me at bob@example.com please", trace=True)
    assert "bob@example.com" not in trace["core"]


def test_cli_plan_runs_without_error(monkeypatch):
    import agent.promptlab as pl
    # clarifying_questions + plan both hit the model; give it a fixed reply and
    # make every console.input() an empty line (skip questions, accept result).
    monkeypatch.setattr(pl, "_client_cfg", lambda: (FakeLLM("Role: dev\nObjective: x"), CFG))
    monkeypatch.setattr("rich.console.Console.input", lambda self, *a, **k: "")
    pl.cli_plan("build me a thing")
