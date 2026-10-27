"""The fixed prompt prefix: scripts/prefix_report.py (size per part, cache breakers) and the byte-stability the server's prompt cache needs.
A model server (llama.cpp / Ollama) re-reads a prompt from the first byte that differs from the previous request."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

from agent import projectmem
from tests.test_codeagent_engine import FakeLLM, _isolated, call, engine, say  # noqa: F401 - _isolated is an autouse fixture

SPEC = importlib.util.spec_from_file_location("prefix_report", Path(__file__).resolve().parent.parent / "scripts" / "prefix_report.py")
PR = importlib.util.module_from_spec(SPEC)
sys.modules["prefix_report"] = PR                     # dataclasses look the module up by name
SPEC.loader.exec_module(PR)


class SchemaLLM(FakeLLM):
    """FakeLLM that also keeps the tool schema of every request."""

    def __init__(self, script):
        super().__init__(script)
        self.schemas: list[str] = []

    def chat_with_tools(self, messages, model, tools, **kw):
        self.schemas.append(json.dumps(tools, sort_keys=True))
        return super().chat_with_tools(messages, model, tools, **kw)


def _run(tmp_path, script):
    llm = SchemaLLM(script)
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "repo" / "a.py").write_text("def f():\n    return 1\n")
    eng, repo, _ = engine(tmp_path, llm)
    eng.turn("add b.py and check a.py")
    return eng, llm


# ── the system message and the request prefix stay byte-identical across steps ──────────────────────

def test_system_message_and_schema_are_byte_identical_across_steps(tmp_path):
    eng, llm = _run(tmp_path, [call("list_dir", path="."), call("read_file", path="a.py"), call("run", command="python3 a.py"),
                               call("write_file", path="b.py", content="x = 1\n"), call("run", command="python3 b.py"),
                               call("search", pattern="def"), say("Done.")])
    assert len(llm.calls) == 7
    first = llm.calls[0]["snapshot"][0]
    assert first["role"] == "system"
    for c in llm.calls[1:]:
        assert c["snapshot"][0] == first and json.dumps(c["snapshot"][0], sort_keys=True) == json.dumps(first, sort_keys=True)
    assert len(set(llm.schemas)) == 1                               # the tool list never changes during a run


def test_every_request_extends_the_previous_one(tmp_path):
    """No message that was already sent is rewritten later (no step counter, hint or timestamp edited into the early context)."""
    bad = {"role": "assistant", "content": "<function=read_file>\n<parameter=path"}
    eng, llm = _run(tmp_path, [call("run", command="python3 -c 'raise SystemExit(3)'"), bad, call("run", command="python3 -c 'raise SystemExit(3)'"),
                               call("write_file", path="b.py", content="x = 1\n"), call("run", command="python3 b.py"), say("Done.")])
    snaps = [c["snapshot"] for c in llm.calls]
    for prev, cur in zip(snaps, snaps[1:]):
        assert cur[:len(prev)] == prev, [i for i in range(len(prev)) if cur[i] != prev[i]]


# ── stage restarts keep the head (system prompt, project memory, repo map, brief card) ─────────────

def test_a_new_stage_after_progress_keeps_the_whole_head_byte_identical(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "a.py").write_text("def f():\n    return 1\n")
    memory = projectmem.ProjectMemory(sqlite3.connect(":memory:"))
    key = str((tmp_path / "repo").resolve())
    memory.add(key, "tests run with pytest -q from the repo root")
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.notes, eng.note_key = memory, key
    first = PR.stage_context(eng, PR.SAMPLE_TASK, "Stage 1: skeleton.", "")
    (repo / "b.py").write_text("def g():\n    return 2\n")                       # progress: a new file, a saved note, steps
    memory.add(key, "the store keeps items in inventory.json")
    eng.session_notes.append("the store keeps items in inventory.json")
    eng.ctx.step += 9
    second = PR.stage_context(eng, PR.SAMPLE_TASK, "Stage 2: report.", "- stage 1 [done]")
    goal_at = next(i for i, m in enumerate(second) if m["content"].startswith("STAGE GOAL"))
    assert goal_at >= 4                                                          # system, memory, repo map, brief card in front of the goal
    assert first[:goal_at] == second[:goal_at]
    assert "b.py" not in second[2]["content"]                                    # the map is the session's snapshot (VERIFIED STATE lists the new file)
    assert "b.py" in second[-1]["content"] and "inventory.json" in "\n".join(m["content"] for m in second if m["content"].startswith("PROJECT NOTES SAVED"))


def test_a_new_session_gets_a_fresh_repo_map(tmp_path):
    from agent.codeagent import session

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.start()
    (repo / "later.py").write_text("x = 1\n")
    session.reset_engine(eng)
    assert any("later.py" in m["content"] for m in eng.ctx.messages if m["content"].startswith("REPO MAP"))


# ── the report ──────────────────────────────────────────────────────────────

def _report(tmp_path):
    repo = PR.make_sample_repo(tmp_path / "sample")
    with PR.isolated_state():
        return PR.build_report(repo)


def test_report_lists_the_parts_sorted_and_finds_no_breaker(tmp_path):
    rep = _report(tmp_path)
    names = [p.name for p in rep.parts]
    for want in ("system prompt", "PROJECT block", "repo map", "brief card", "stage goal", "stage notes", "verified state", "project memory", "tool run", "tool read_file"):
        assert want in names, want
    tokens = [p.tokens for p in rep.parts]
    assert tokens == sorted(tokens, reverse=True) and tokens[0] > 100
    assert rep.breakers == []
    text = PR.render(rep, top=5)
    assert "PREFIX-CACHE BREAKERS: none found" in text and text.count("\n") < 20 and "smaller parts" in text


def test_report_flags_a_timestamp_inside_the_system_message(tmp_path, monkeypatch):
    import time

    from agent.codeagent import engine as E

    monkeypatch.setattr(E, "SYSTEM", E.SYSTEM + "\nIt is now " + "{}".format(1))                       # a constant is fine ...
    assert _report(tmp_path).breakers == []
    real = E.CodeEngine.start

    def stamped(self):                                                                                  # ... a clock reading is not
        real(self)
        self._head = None
        self.ctx.messages[0] = {**self.ctx.messages[0], "content": self.ctx.messages[0]["content"] + f"\nnow: {int(time.time())}"}

    monkeypatch.setattr(E.CodeEngine, "start", stamped)
    rep = _report(tmp_path / "x")
    assert any("system prompt (message 0) differs between two builds" in b for b in rep.breakers)


def test_report_flags_a_stage_restart_breaker_and_step_counters(tmp_path):
    msgs = [{"role": "system", "content": "SYSTEM text at step 12"}, {"role": "system", "content": "STAGE GOAL (pinned)\nx"}]
    assert any("step counter" in w for w in PR.scan_early(msgs))
    a = [{"role": "system", "content": "s"}, {"role": "system", "content": "REPO MAP a"}, {"role": "system", "content": "STAGE GOAL 1"}]
    b = [{"role": "system", "content": "s"}, {"role": "system", "content": "REPO MAP b"}, {"role": "system", "content": "STAGE GOAL 2"}]
    warn, summary = PR.check_restart(a, b)
    assert len(warn) == 1 and "repo map" in warn[0] and "#1 of 3" in summary


def test_report_cli_prints_and_exits_zero(tmp_path, capsys):
    assert PR.main(["--top", "4"]) == 0
    out = capsys.readouterr().out
    assert "FIXED PROMPT PREFIX" in out and "system prompt" in out
