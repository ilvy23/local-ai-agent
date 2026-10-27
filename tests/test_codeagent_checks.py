"""Items 2 + 3 (D3, D4): checks as first-class objects - command / python / banned / rubric -, supervisor checks that the model
cannot touch, mark_done gated by checks, the bounded final gate with a NOT VERIFIED label, the sentence-level text helpers, and
the reviewer's deterministic rules (real quotes, one JSON retry, then 'unreviewed')."""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from agent.codeagent import checks as K, reviewer as R, textcheck as T

from tests.test_codeagent_engine import FakeLLM, call, engine, say  # noqa: F401


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


# ── textcheck (D4) ──────────────────────────────────────────────────────────

def test_the_same_sentence_wrapped_differently_is_a_duplicate():
    a = "He pours the tea with great ceremony.\nThen he looks away."
    b = "He pours the tea\nwith great ceremony. A cat sleeps."
    assert a.splitlines()[0] not in b.splitlines()                       # a line-level test sees nothing
    dups = T.duplicates({"a": a, "b": b})
    assert list(dups) == ["he pours the tea with great ceremony"] and dups["he pours the tea with great ceremony"] == {"a", "b"}
    assert T.overlap_problems({"steady": a, "thin": b})
    assert T.repeats_inside({"x": "Sit down now. Sit   down now!"}) == [("x", "sit down now", 2)]


def test_banned_hits_and_flat():
    assert T.banned_hits("It is hollow. HOLLOW!", [r"\bhollow\b", "for a while"]) == [(r"\bhollow\b", 2)]
    assert T.flat({"a": ["x", ("y", {"k": "z"})], "n": 3}) == ["x", "y", "z"]


def test_finish_exits_with_the_verdict(capsys):
    with pytest.raises(SystemExit) as e:
        T.finish("C", ["one problem"])
    assert e.value.code == 1 and "C: FAIL (1 problems)" in capsys.readouterr().out


# ── running checks ──────────────────────────────────────────────────────────

def runner(tmp_path, review=None, allow=lambda c: True):
    from agent.codeagent import profile as PR
    from agent.codeagent.context import ContextManager
    from agent.codeagent.eventlog import EventLog
    from agent.codeagent.tools import CodeTools

    ctx = ContextManager(None, {"models": {"background": "m"}}, 32768, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    tools = CodeTools(tmp_path, PR.detect(tmp_path), ctx)
    return K.Runner(tmp_path, exec=lambda c, t: tools._exec(c, t), allow=allow, review=review)


def test_command_python_banned_and_rubric_checks(tmp_path):
    (tmp_path / "lena.py").write_text('OPEN = {"a": "He pours tea. He is hollow.", "b": "He pours tea."}\n')
    r = runner(tmp_path, review=lambda crit, out: {"status": "pass" if "pours" in out else "fail", "evidence": ["He pours tea."], "problems": []})
    assert K.run(K.Check("c1", "true", "command", {"command": "true"}), r)["status"] == "pass"
    assert K.run(K.Check("c2", "false", "command", {"command": "echo boom; false"}), r)["status"] == "fail"
    script = tmp_path / ".scratch" / "checks" / "dup.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent("""
        import lena
        from agent.codeagent import textcheck as tc
        tc.finish("DUP", tc.duplicate_problems(lena.OPEN))
    """))
    res = K.run(K.Check("p", "no shared sentences", "python", {"path": ".scratch/checks/dup.py"}), r)
    assert res["status"] == "fail" and "DUP: FAIL" in res["output"] and "he pours tea" in res["output"]
    res = K.run(K.Check("b", "no gloom", "banned", {"patterns": [r"\bhollow\b"], "objects": ["lena:OPEN"]}), r)
    assert res["status"] == "fail" and "lena:OPEN" in res["output"]
    assert K.run(K.Check("b2", "files", "banned", {"patterns": ["ledger"], "files": ["*.py"]}), r)["status"] == "pass"
    res = K.run(K.Check("r", "cosy", "rubric", {"criterion": "he pours tea", "scenario": ["python3 -c \"import lena; print(lena.OPEN['b'])\""]}), r)
    assert res["status"] == "pass" and "evidence" in res["output"]
    assert K.run(K.Check("m", "missing", "python", {"path": "nope.py"}), r)["status"] == "error"
    blocked = runner(tmp_path, allow=lambda c: False)
    assert K.run(K.Check("c3", "x", "command", {"command": "true"}), blocked)["status"] == "skipped"


def test_validate_rejects_incomplete_specs():
    assert K.validate("command", {}) and K.validate("banned", {"patterns": ["x"]}) and K.validate("rubric", {"criterion": "x"})
    assert K.validate("weird", {}) and K.validate("python", {"path": "a.py"}) is None


# ── supervisor checks ───────────────────────────────────────────────────────

def sup_dir(repo):
    d = K.checks_dirs(repo)[0]
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_supervisor_checks_are_picked_up_and_read_only(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    d = sup_dir(repo)
    (d / "lena_check.py").write_text("print('LENA CHECK: PASS')\n")
    (d / "more.json").write_text(json.dumps([{"id": "tests", "kind": "command", "spec": {"command": "true"}},
                                             {"id": "bad", "kind": "banned", "spec": {}}]))
    ids = {c.id: c for c in eng.checks.all()}
    assert set(ids) == {"lena_check", "tests"} and all(c.readonly for c in ids.values())
    assert any("bad" in w for w in eng.checks.warnings)
    out = eng.registry.get("add_check").handler(id="lena_check", kind="command", command="true", description="mine")
    assert "read-only" in out
    res = eng.execute("write_file", {"path": str(d / "lena_check.py"), "content": "import sys; sys.exit(0)\n"})
    assert res.startswith(("NOT RUN", "BLOCKED")) and "PASS" in (d / "lena_check.py").read_text()      # outside the repo: never written


def test_final_gate_sends_the_model_back_until_checks_pass(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="X = 1\n"),
                   say("Done, all good."),
                   call("write_file", path="a.py", content="X = 2\n"),
                   say("Fixed: X is 2.")])
    eng, repo, _ = engine(tmp_path, llm)
    (sup_dir(repo) / "x2.sh").write_text("grep -q 'X = 2' a.py || { echo 'X must be 2'; exit 1; }\n")
    res = eng.turn("set X")
    assert res.text == "Fixed: X is 2." and res.not_verified == []
    back = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and m["content"].startswith("CHECKS FAILED")]
    assert len(back) == 1 and "X must be 2" in back[0] and "x2" in back[0]
    assert [e["status"] for e in eng.events.recent if e["kind"] == "check"] == ["fail", "pass"]


def test_a_check_that_never_passes_is_bounded_and_labelled(tmp_path):
    llm = FakeLLM([call("write_file", path="a.py", content="X = 1\n")] + [say("Done.")] * 5)
    eng, repo, _ = engine(tmp_path, llm)
    (sup_dir(repo) / "never.sh").write_text("echo nope; exit 1\n")
    res = eng.turn("set X")
    assert res.text.startswith("NOT VERIFIED: never (fail)") and res.verified is False
    assert len([m for m in eng.ctx.messages if str(m["content"]).startswith("CHECKS FAILED")]) == K.CHECK_RETURNS


def test_failure_message_reshows_the_brief_card(tmp_path):
    from tests.test_codeagent_brief import LENA

    llm = FakeLLM([say("Done.")] * 5)
    eng, repo, _ = engine(tmp_path, llm)
    (sup_dir(repo) / "never.sh").write_text("exit 1\n")
    eng.turn(LENA)
    back = next(m["content"] for m in eng.ctx.messages if str(m["content"]).startswith("CHECKS FAILED"))
    assert eng.brief() and eng.brief() in back


def test_mark_done_needs_the_items_checks_to_pass(tmp_path):
    from agent import plan as plan_mod

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.plan.plan = plan_mod.Plan("t", [plan_mod.Item(1, "X is 2"), plan_mod.Item(2, "docs")])
    (repo / "a.py").write_text("X = 1\n")
    out = eng.registry.get("add_check").handler(id="x2", kind="command", command="grep -q 'X = 2' a.py", description="X is 2", items=[1])
    assert "added" in out and "FAIL" in out
    md = eng.registry.get("mark_done").handler
    out = md(ids=[1, 2], evidence="I am sure it works")
    assert "item 1 NOT confirmed" in out and eng.plan.plan.items[1].done and not eng.plan.plan.items[0].done
    (repo / "a.py").write_text("X = 2\n")
    out = md(id=1, evidence="ok")
    assert eng.plan.plan.items[0].done and "harness: checks x2 passed" in eng.plan.plan.items[0].evidence


def test_model_checks_survive_a_resume(tmp_path):
    from agent.codeagent import session as S, sessionstore as SS

    eng, repo, _ = engine(tmp_path, FakeLLM([say("ok")]))
    eng.session_id = SS.new_id()
    eng.registry.get("add_check").handler(id="t", kind="command", command="true", description="truth")
    eng.turn("start")
    S.save_session(eng)
    eng2, _r, _o = engine(tmp_path, FakeLLM([]))
    S.resume_into(eng2, SS.latest(eng.root))
    c = eng2.checks.get("t")
    assert c is not None and c.status() == "pass" and not c.readonly


def test_rubric_is_judged_again_only_after_a_change(tmp_path):
    seen = []
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng._review = lambda crit, out: seen.append(out) or {"status": "fail", "evidence": [], "problems": ["dull"]}
    eng.checks.add(K.Check("tone", "warm", "rubric", {"criterion": "warm", "scenario": ["echo hi"]}))
    eng.run_checks(quiet=True)
    eng.run_checks(quiet=True)
    assert len(seen) == 1
    eng._progress += 1
    eng.run_checks(quiet=True)
    assert len(seen) == 2 and "$ echo hi\nhi" in seen[0]


# ── reviewer (item 3) ───────────────────────────────────────────────────────

class ReviewLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat_with_tools(self, messages, model, tools, num_ctx=None, think=None, extra_options=None):
        self.calls.append({"messages": messages, "think": think, "opts": extra_options, "num_ctx": num_ctx, "tools": tools})
        return {"content": self.replies.pop(0), "_prompt_eval_count": 900}


OUTPUT = "Lena: T-tch, it's not like I baked this for you!\nLena: Hmph. Sit down, dummy."


def test_reviewer_pass_needs_a_real_quote():
    llm = ReviewLLM(['{"pass": true, "evidence": ["it\'s not like I baked this for you"], "problems": []}'])
    res = R.review(llm, "m", "BRIEF CARD ...", "bossy voice", OUTPUT, num_ctx=98304)
    assert res["status"] == "pass" and res["evidence"]
    c = llm.calls[0]
    assert c["think"] is False and c["tools"] == [] and c["num_ctx"] == 98304 and c["opts"]["num_predict"] == R.REVIEW_PREDICT
    assert "USER'S BRIEF" in c["messages"][1]["content"] and len(c["messages"]) == 2      # fresh context: no history
    llm = ReviewLLM(['{"pass": true, "evidence": ["he is very cute and warm"], "problems": []}'])
    assert R.review(llm, "m", "", "cute", OUTPUT)["status"] == "unreviewed"               # an invented quote is not evidence


def test_reviewer_retries_invalid_json_once_then_gives_up():
    llm = ReviewLLM(["I think it passes!", '```json\n{"pass": false, "evidence": ["Hmph. Sit down, dummy."], "problems": ["too short"]}\n```'])
    res = R.review(llm, "m", "", "long scenes", OUTPUT)
    assert res["status"] == "fail" and res["problems"] == ["too short"] and len(llm.calls) == 2
    assert R.review(ReviewLLM(["nope", "still nope"]), "m", "", "x", OUTPUT)["status"] == "unreviewed"


def test_reviewer_input_is_bounded():
    llm = ReviewLLM(['{"pass": false, "evidence": [], "problems": []}'])
    R.review(llm, "m", "B" * 20000, "C" * 5000, "O" * 200000)
    assert len(llm.calls[0]["messages"][1]["content"]) < 16000                        # ~5k tokens in


def test_reviewer_unreachable_is_unreviewed():
    class Down:
        def chat_with_tools(self, *a, **k):
            raise OSError("connection refused")
    assert R.review(Down(), "m", "", "x", "y")["status"] == "unreviewed"


def test_python_check_uses_the_agents_interpreter(tmp_path):
    cmd = K._python_cmd(tmp_path, "x.py")
    assert sys.executable in cmd and str(tmp_path) in cmd
