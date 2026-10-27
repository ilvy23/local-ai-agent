"""The verified-state block (agent/codeagent/verified.py): the harness measures what exists, what passes, what fails and what is open,
so a resumed session starts working instead of re-reading 80 files. No model call anywhere."""

from __future__ import annotations

import io
import sys
import textwrap

from rich.console import Console

from agent.codeagent import checks as checks_mod, context, session as S, sessionstore as SS, verified
from agent.plan import Item, Plan

from tests.test_codeagent_engine import CFG, FakeLLM, engine, say
from tests.test_codeagent_sessions import _isolated, first_turn, new_engine  # noqa: F401  (the autouse isolation fixture)

PY = sys.executable
HEADER = "VERIFIED STATE - measured by the harness just now, not remembered. Trust it; do not re-read files to confirm it."


def cfg(**coding):
    return {**CFG, "coding": {**CFG["coding"], **coding}}


def note(eng, repo, rel, text, action="wrote", step=1):
    f = repo / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    eng.ctx.ledger.note_file(rel, text, step, action)


def pytest_cmd(eng):
    eng.profile.test_cmd = f"{PY} -m pytest -q -x -p no:cacheprovider"        # the default carries -x: the block must still list every failure


def full_repo(tmp_path):
    eng, repo, _out = engine(tmp_path, FakeLLM([]))
    eng.config = cfg()
    note(eng, repo, "calc.py", "def add(a, b):\n    return a + b\n\n\nclass Calc:\n    pass\n", "edited", 3)
    note(eng, repo, "test_ok.py", "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n", "wrote", 4)
    note(eng, repo, "test_bad.py", "def test_wrong():\n    assert 1 == 2, 'one is not two'\n", "wrote", 5)
    note(eng, repo, "broken.py", "def f(:\n    pass\n", "wrote", 6)
    pytest_cmd(eng)
    eng.checks.add(checks_mod.Check("exit3", "must exit zero", "command", {"command": "echo boom on stdout; exit 3"}))
    eng.checks.add(checks_mod.Check("fine", "always fine", "command", {"command": "true"}))
    eng.checks.add(checks_mod.Check("tone", "reads well", "rubric", {"criterion": "reads well", "scenario": ["echo hi"]},
                                    last_result={"status": "fail", "output": "too dry"}))
    eng.plan.plan = Plan("t", [Item(1, "calc.py has add", True), Item(2, "calc.py has subtract with a test"), Item(3, "README explains usage")])
    eng.last_stopped = "step budget"
    return eng, repo


def test_all_sections_show_the_measured_facts(tmp_path):
    eng, repo = full_repo(tmp_path)
    st = verified.build(eng)
    t = st.text
    assert t.startswith(HEADER) and len(t) <= verified.MAX_CHARS and st.seconds >= 0
    assert "1 passed, 1 failed" in t and "test_bad.py::test_wrong" in t and "one is not two" in t
    assert "exit3: FAIL - boom on stdout" in t
    assert "fine: PASS" in t and "tone: FAIL (rubric, stored verdict, not re-judged) - too dry" in t
    assert "broken.py:1" in t and "SyntaxError" in t
    assert "1 of 3 confirmed" in t and "2  calc.py has subtract with a test" in t and "3  README explains usage" in t
    assert "calc.py  6 lines  edited (step 3)" in t and "test_bad.py" in t
    assert "add:1" in t and "Calc:5" in t                                    # the symbol map: top-level names with their line
    assert "step budget" in t
    assert set(st.sections) >= {"TESTS", "CHECKS", "STATIC", "OPEN ITEMS", "FILES", "SYMBOLS", "LAST STOP"}
    assert st.sections["TESTS"]["passed"] == 1 and st.sections["TESTS"]["failed"] == 1
    assert context.new_markers(None, t) == [] and "[..." not in t and "[…" not in t and "|" not in t


def test_a_project_without_tests_says_so(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.config = cfg()
    eng.profile.test_cmd = None
    st = verified.build(eng)
    assert "no test command detected" in st.text
    assert "no acceptance list" in st.text


def test_tests_not_run_when_asked_or_configured(tmp_path):
    eng, repo = full_repo(tmp_path)
    assert "tests not run" in verified.build(eng, run_tests=False).text
    eng.config = cfg(verified_tests=False)
    st = verified.build(eng)
    assert "tests not run" in st.text and "1 passed" not in st.text
    assert "passed" in verified.build(eng, run_tests=True).text                # an explicit argument beats the config


def test_a_hanging_test_command_is_cut_by_the_configured_timeout(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.config = cfg(verified_timeout_s=1)
    (repo / "hang.py").write_text("import time\ntime.sleep(60)\n")
    eng.profile.test_cmd = f"{PY} hang.py"
    st = verified.build(eng)
    assert "tests timed out after 1s" in st.text
    assert st.seconds < 20


def test_no_model_call_is_made(tmp_path):
    llm = FakeLLM([])                                                          # any chat_with_tools would raise IndexError
    eng, repo, _ = engine(tmp_path, llm)
    eng.config = cfg()
    note(eng, repo, "a.py", "x = 1\n")
    eng.profile.test_cmd = None
    st = verified.build(eng)
    assert st.text.startswith(HEADER) and llm.calls == []


def test_the_block_stays_under_the_cap_and_drops_the_low_priority_sections_first(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.config = cfg()
    for i in range(300):
        note(eng, repo, f"pkg/mod_{i:03}.py", f"def func_{i}():\n    return {i}\n\n\nclass Thing{i}:\n    pass\n", "wrote", i)
    (repo / "boom.py").write_text("print('x')\n" + "".join(
        f"print('FAILED tests/test_many.py::test_case_{i} - AssertionError: assert {i} == {i + 1} and a long long tail ' + 'y' * 40)\n"
        for i in range(5000)) + "print('5000 failed in 1.00s')\nraise SystemExit(1)\n")
    eng.profile.test_cmd = f"{PY} boom.py"
    eng.plan.plan = Plan("t", [Item(i, f"acceptance item number {i} " + "with a long description " * 8) for i in range(1, 31)])
    for i in range(12):
        eng.checks.add(checks_mod.Check(f"chk{i}", "d", "command", {"command": f"echo failing line number {i} " + "z" * 150 + "; exit 1"}))
    st = verified.build(eng)
    assert len(st.text) <= verified.MAX_CHARS
    assert "5000 failed" in st.text and "test_case_0 " in st.text and "test_case_7 " in st.text and "test_case_8 " not in st.text
    assert "acceptance item number 1 " in st.text and "0 of 30 confirmed" in st.text
    assert "TESTS:" in st.text and "OPEN ACCEPTANCE ITEMS:" in st.text

    # with a smaller budget the order of sacrifice is visible: SYMBOLS goes first, FILES next, TESTS and OPEN ITEMS stay
    old = verified.MAX_CHARS
    verified.MAX_CHARS = 7500
    try:
        small = verified.build(eng)
    finally:
        verified.MAX_CHARS = old
    assert len(small.text) <= 7500
    assert small.dropped and small.dropped == ["LAST STOP", "SYMBOLS", "FILES"][: len(small.dropped)]       # lowest priority first, never TESTS / OPEN ITEMS
    assert "SYMBOL MAP" not in small.text and "LAST STOP:" not in small.text
    assert "TESTS:" in small.text and "OPEN ACCEPTANCE ITEMS:" in small.text
    assert "left out for size: " + ", ".join(small.dropped) in small.text
    assert context.new_markers(None, small.text) == []


def test_the_section_order_of_dropping_is_lowest_priority_first():
    secs = [(n, "x" * 100) for n in ("TESTS", "CHECKS", "STATIC", "OPEN ITEMS", "FILES", "SYMBOLS", "LAST STOP")]
    kept, dropped = verified.fit(secs, 450)
    assert [n for n, _ in kept] == ["TESTS", "CHECKS", "STATIC", "OPEN ITEMS"] and dropped == ["LAST STOP", "SYMBOLS", "FILES"]


def test_resume_appends_the_verified_state_and_the_next_turn_sends_it_once(tmp_path):
    eng, repo, res = first_turn(tmp_path)
    saved_msgs = [dict(m) for m in eng.ctx.messages]
    llm2 = FakeLLM([say("ok, a.py prints hi.")])
    eng2, _r, out2 = new_engine(tmp_path, llm2)
    meta = SS.latest(repo)
    info = S.resume_into(eng2, meta)
    assert info
    assert "SESSION RESUMED" in eng2.resume_note and "VERIFIED STATE" in eng2.resume_note
    assert "a.py" in eng2.resume_note and "no test command detected" in eng2.resume_note
    assert "computing verified state" in out2.getvalue()
    eng2.turn("anything left?")
    sent = llm2.calls[0]["snapshot"]
    carrying = [m for m in sent if m["role"] == "system" and "VERIFIED STATE" in str(m["content"])]
    assert len(carrying) == 1 and carrying[0] is not None and sent.index(carrying[0]) == len(saved_msgs)
    assert eng2.resume_note is None


def test_resume_without_the_block_when_configured(tmp_path):
    eng, repo, res = first_turn(tmp_path)
    llm2 = FakeLLM([])
    eng2, _r, out2 = new_engine(tmp_path, llm2)
    eng2.config = cfg(verified_on_resume=False)
    S.resume_into(eng2, SS.latest(repo))
    assert "SESSION RESUMED" in eng2.resume_note and "VERIFIED STATE" not in eng2.resume_note


def test_state_slash_command_prints_the_block(tmp_path):
    eng, repo = full_repo(tmp_path)
    buf = io.StringIO()
    con = Console(file=buf, width=200, force_terminal=False)
    assert S.handle_command(con, eng, "/state") is True
    out = buf.getvalue()
    assert "VERIFIED STATE" in out and "1 passed, 1 failed" in out
    assert "/state" in S.HELP and "agent code state" in S.HELP


def test_cli_state_works_with_and_without_a_saved_session(tmp_path):
    from typer.testing import CliRunner

    from agent import main

    eng, repo, _res = first_turn(tmp_path)                                     # a folder with a saved session
    r = CliRunner().invoke(main.app, ["code", "state", "--folder", str(repo)])
    assert r.exit_code == 0, r.output
    assert "VERIFIED STATE" in r.output and "a.py" in r.output

    bare = tmp_path / "bare"                                                   # and one the agent never saw
    bare.mkdir()
    (bare / "x.py").write_text("def hello():\n    return 1\n")
    r2 = CliRunner().invoke(main.app, ["code", "state", "--folder", str(bare)])
    assert r2.exit_code == 0, r2.output
    assert "VERIFIED STATE" in r2.output and "x.py" in r2.output and "hello" in r2.output



def test_failures_known_red_before_the_stage_are_labelled_not_yours(tmp_path):
    """Seen live: 2 reds that belong to later parts made the model investigate them instead of finishing its own part."""
    from agent.codeagent import stages as ST
    eng, repo = full_repo(tmp_path)
    plan = ST.StagePlan([ST.Stage(1, "part 1", "g1"), ST.Stage(2, "part 2", "g2"), ST.Stage(3, "part 3", "g3")], 0, "headings", "task")
    eng.stage_plan = plan
    plan.baseline_red = ["test_bad.py::test_wrong"]
    t = verified.build(eng).text
    assert "already red before this stage" in t and "test_bad.py::test_wrong" in t
    assert "(already red)" in t
    plan.baseline_red = []                                           # a NEW failure is shown plainly
    assert "already red" not in verified.build(eng).text
    plan.baseline_red = ["test_bad.py::test_wrong"]
    plan.current = 2                                                 # the last stage is strict: nothing is excused
    assert "already red" not in verified.build(eng).text
    eng.stage_plan = None                                            # outside a staged run: unchanged
    assert "already red" not in verified.build(eng).text
