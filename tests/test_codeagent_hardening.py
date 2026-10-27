"""Hardening of the coding engine: streaming + degenerate-output stop, model errors, loop guard on real message shapes,
bounded gates, safe process handling, atomic / protected writes, line endings."""

from __future__ import annotations

import io
import json
import os
import stat
import time

import httpx
import pytest
from rich.console import Console

from agent import llm
from agent.codeagent import context as C, policy as P, tools as T
from agent.codeagent.engine import CodeEngine, repeating
from agent.codeagent.eventlog import EventLog
from agent.llm import GenerationAborted

from tests.test_codeagent_engine import CFG, FakeLLM, call, say


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CODE_POLICY", str(tmp_path / "policy.json"))
    monkeypatch.setenv("AGENT_CODE_EVENTS", str(tmp_path / "events.jsonl"))
    from agent import repoaccess

    monkeypatch.setattr(repoaccess, "greenlist", lambda: [])
    repoaccess.reset_session()


def engine(tmp_path, llm_, mode="auto", approve=None, **pol):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    P.save_repo(repo, mode=mode, **pol)
    out = io.StringIO()
    eng = CodeEngine(repo, CFG, llm_, Console(file=out, width=140, force_terminal=False), approve_fn=approve, interactive=False,
                     events=EventLog("t", tmp_path / "ev.jsonl"))
    return eng, repo, out


# ── streaming and degenerate output ─────────────────────────────────────────

def _stream_client(chunks):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, text="\n".join(json.dumps(c) for c in chunks))

    return llm.OllamaClient(transport=httpx.MockTransport(handler)), seen


def _piece(text="", done=False, **extra):
    return {"message": {"role": "assistant", "content": text}, "done": done, **extra}


def test_stream_passes_think_and_options_and_reports_timing_and_done_reason():
    client, seen = _stream_client([_piece("hi"), _piece(done=True, prompt_eval_count=5, eval_count=2, eval_duration=10**9,
                                                        prompt_eval_duration=2 * 10**8, done_reason="length")])
    msg = client.chat_with_tools_stream([], "m:1", [], num_ctx=8192, think=True, extra_options={"num_predict": 7, "temperature": 0.25})
    assert seen["payload"]["think"] is True and seen["payload"]["options"]["num_predict"] == 7
    assert msg["_eval_ns"] == 10**9 and msg["_prompt_eval_ns"] == 2 * 10**8 and msg["_done_reason"] == "length"


def test_stream_error_line_after_the_header_raises():
    client, _ = _stream_client([_piece("par"), {"error": "llama runner process has terminated"}])
    with pytest.raises(httpx.HTTPStatusError, match="runner process"):
        client.chat_with_tools_stream([], "m:1", [])


def test_a_callback_can_abort_the_stream_and_gets_the_partial_reply():
    client, _ = _stream_client([_piece("a"), _piece("b"), _piece("c"), _piece(done=True)])
    seen = []

    def on_content(p):
        seen.append(p)
        if len(seen) == 2:
            raise GenerationAborted("content repeats itself")

    msg = client.chat_with_tools_stream([], "m:1", [], on_content=on_content)
    assert msg["content"] == "ab" and msg["_aborted"] == "content repeats itself" and seen == ["a", "b"]


def test_repetition_detector_flags_loops_but_not_long_legit_code():
    assert repeating("start\n" + "    self.items.append(item)\n" * 400)
    assert repeating("Thinking. " + "Wait, let me reconsider the approach. " * 300)
    legit = "".join(f"def handler_{i}(state):\n    return state.step({i}, '{i * 7}')\n\n" for i in range(800))
    assert not repeating(legit) and not repeating("x" * 100)
    table = "".join(f'    ("room_{i}", "a cold room", {i}, {i % 7}),\n' for i in range(600))
    assert not repeating(table)


class StreamLLM(FakeLLM):
    """Streams each scripted reply piece by piece through the engine's callbacks, like the real client."""

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, think=None, extra_options=None):
        self.calls.append({"n": len(messages), "think": think, "opts": extra_options, "snapshot": [dict(m) for m in messages]})
        step = dict(self.script.pop(0))
        text, got = step.get("content", ""), []
        try:
            for i in range(0, len(text), 40):
                got.append(text[i:i + 40])
                on_content(text[i:i + 40])
        except GenerationAborted as stop:
            return {"role": "assistant", "content": "".join(got), "_aborted": str(stop)}
        return step


def test_degenerate_reply_is_stopped_discarded_and_rerolled(tmp_path):
    loop = '{"tool": "write_file", "arguments": {"path": "a.py", "content": "' + "x = compute(a, b)\\n" * 3000
    llm_ = StreamLLM([say(loop), say(loop), call("list_dir", path="."), say("Listed the folder.")])
    eng, repo, out = engine(tmp_path, llm_)
    res = eng.turn("look")
    assert res.text == "Listed the folder." and "repeating itself" in out.getvalue()
    assert not any("compute(a, b)" in str(m.get("content")) for m in eng.ctx.messages)          # the garbage never enters the history
    notes = [m["content"] for m in eng.ctx.messages if m["role"] == "system" and "degenerated" in m["content"]]
    assert len(notes) == 1                                                  # first time: silent re-roll; second time: a note
    assert len([r for r in eng.events.recent if r["kind"] == "degenerate"]) == 2
    assert len(llm_.calls[0]["snapshot"]) == len(llm_.calls[1]["snapshot"])


def test_a_long_legit_streamed_write_is_not_stopped(tmp_path):
    body = "".join(f"def f{i}(x):\n    return x + {i}\n\n" for i in range(1500))
    llm_ = StreamLLM([call("write_file", path="big.py", content=body), say("Wrote big.py.")])
    llm_.script[0] = {"role": "assistant", "content": json.dumps({"tool": "write_file", "arguments": {"path": "big.py", "content": body}})}
    eng, repo, out = engine(tmp_path, llm_)
    res = eng.turn("write big.py")
    assert (repo / "big.py").read_text() == body and res.text == "Wrote big.py." and "repeating" not in out.getvalue()


def test_cut_off_reply_is_reported_not_taken_as_an_answer(tmp_path):
    cut = {"role": "assistant", "content": '{"tool": "write_file", "arguments": {"path": "a.py", "content": "def f():', "_done_reason": "length",
           "_eval_count": 32768}
    llm_ = FakeLLM([cut, say("Gave up, told the user.")])
    eng, repo, out = engine(tmp_path, llm_)
    res = eng.turn("write a.py")
    assert res.text == "Gave up, told the user." and "output limit" in out.getvalue()
    assert any("cut off after 32768 tokens" in str(m["content"]) for m in eng.ctx.messages if m["role"] == "system")


# ── model errors ────────────────────────────────────────────────────────────

class FailingLLM(FakeLLM):
    def __init__(self, script, errors):
        super().__init__(script)
        self.errors = list(errors)

    def chat_with_tools(self, *a, **k):
        if self.errors:
            exc = self.errors.pop(0)
            if exc is not None:
                raise exc
        return super().chat_with_tools(*a, **k)


def _http_error(status, text):
    req = httpx.Request("POST", "http://x/api/chat")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(status, text=text, request=req))


def test_an_unclassified_http_error_is_retried_once_with_the_same_state(tmp_path):
    llm_ = FailingLLM([say("Done.")], [httpx.DecodingError("bad gzip")])           # a dropped connection waits for the server instead (test_codeagent_robust)
    eng, repo, out = engine(tmp_path, llm_)
    res = eng.turn("say done")
    assert res.text == "Done." and "retrying once" in out.getvalue() and eng.num_ctx == 32768


def test_repeated_failures_stop_clearly_and_keep_the_conversation(tmp_path):
    llm_ = FailingLLM([call("list_dir", path="."), say("never")], [None, _http_error(500, '{"error":"llama runner exited"}'),
                                                                     httpx.RemoteProtocolError("peer closed"), httpx.ReadError("reset")])
    eng, repo, out = engine(tmp_path, llm_)          # no server_url on this fake: the three attempts follow at once
    res = eng.turn("look")
    assert res.stopped == "error" and "failed after 3 attempts" in out.getvalue() and eng.num_ctx == 32768      # a plain 500 is not "out of memory"
    assert any(m["role"] == "tool" for m in eng.ctx.messages)                # the work so far is still in the context


def test_only_a_real_out_of_memory_error_shrinks_the_window(tmp_path):
    llm_ = FailingLLM([say("Done.")], [_http_error(500, '{"error":"cudaMalloc failed: out of memory"}')])
    eng, repo, out = engine(tmp_path, llm_)
    assert eng.turn("x").text == "Done." and eng.num_ctx == 24576 and "ran out of memory" in out.getvalue()


def test_an_internal_error_ends_the_prompt_not_the_session(tmp_path, monkeypatch):
    eng, repo, out = engine(tmp_path, FakeLLM([call("list_dir", path=".")]))
    monkeypatch.setattr(eng, "_run_call", lambda c: (_ for _ in ()).throw(RuntimeError("bug")))
    res = eng.turn("look")
    assert res.stopped == "error" and "internal error: RuntimeError: bug" in out.getvalue()


def test_an_unjudgeable_path_is_blocked_not_a_crash(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    monkeypatch.setattr(eng, "decide", lambda n, a: (_ for _ in ()).throw(OSError(36, "File name too long")))
    assert eng.execute("read_file", {"path": "x" * 5000}).startswith("BLOCKED: could not check this call")


def test_more_than_four_calls_are_not_silently_dropped(tmp_path):
    many = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "list_dir", "arguments": {"path": f"d{i}"}}} for i in range(6)]}
    eng, repo, _ = engine(tmp_path, FakeLLM([many, say("ok")]))
    eng.turn("look")
    assert any("Only the first 4 of your 6 tool calls" in str(m["content"]) for m in eng.ctx.messages)


# ── loop guard on the real message shapes ───────────────────────────────────

def _play(eng, command, output, secs):
    eng.ctx.add({"role": "assistant", "content": json.dumps({"tool": "run", "arguments": {"command": command}})})
    eng.ctx.add({"role": "tool", "content": f"exit code 0  ({secs}s)\n{output}"})
    eng._loop_guard("run", {"command": command}, f"exit code 0  ({secs}s)\n{output}")


def test_nearly_identical_playthroughs_are_caught_and_collapsed(tmp_path):
    """The live failure: the same piped playthrough with two inputs swapped, 6.6 s / 8.1 s timings, the same banner and ending."""
    eng, repo, out = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 30
    banner = "~" * 60 + "\n  H A R B O U R\n  a harbour story\n" + "you stand in the square.\n" * 40
    a = "cd .scratch/harbour_test && printf 'status\\njournal\\nask ella lake\\nask orin canal\\ntalk\\n1\\nquit\\n' | python3 main.py 2>&1 | head -200"
    b = "cd .scratch/harbour_test && printf 'journal\\nstatus\\nask ella lake\\nask orin canal\\ntalk\\n1\\nquit\\n' | python3 main.py 2>&1 | head -200"
    _play(eng, a, banner + "the bell rings. (end)", 6.6)
    eng.ctx.add({"role": "assistant", "content": json.dumps({"tool": "search", "arguments": {"pattern": "asked_miller_cost"}})})
    eng.ctx.add({"role": "tool", "content": "202: if s.has('miller_paid')"})
    eng._loop_guard("search", {"pattern": "asked_miller_cost"}, "202: if s.has('miller_paid')")
    _play(eng, b, banner + "the bell rings. (end)", 8.1)
    assert not [m for m in eng.ctx.messages if m["role"] == "system"]
    _play(eng, a, banner + "the bell rings. (end) ", 8.1)
    hints = [m["content"] for m in eng.ctx.messages if m["role"] == "system"]
    assert len(hints) == 1 and "nearly identical" in hints[0]
    runs = [m for m in eng.ctx.messages if m["role"] == "assistant" and '"run"' in m["content"]]
    assert len(runs) == 1 and eng.ctx.messages[-1]["role"] == "system"                   # earlier copies removed, the search stays
    assert any('"search"' in m["content"] for m in eng.ctx.messages if m["role"] == "assistant")
    for i in range(5):
        eng.ctx.step += 1
        _play(eng, a if i % 2 else b, banner + "the bell rings. (end)", 7.0)
    assert eng._abort_loop >= 4


def test_a_three_step_cycle_is_a_loop_and_a_write_starts_fresh(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 5
    cycle = [("run", {"command": "python3 main.py"}, "exit code 0\nsame"), ("search", {"pattern": "x"}, "1: x"),
             ("run", {"command": "cat save.json"}, "exit code 0\n{}")]
    for _ in range(2):
        for name, args, res in cycle:
            eng._loop_guard(name, args, res)
    assert not [m for m in eng.ctx.messages if m["role"] == "system"]
    eng._loop_guard(*cycle[0])
    assert [m for m in eng.ctx.messages if m["role"] == "system"]
    eng._loop_guard("edit_file", {"path": "main.py"}, "edited main.py: 1 edit(s) applied (exact). Syntax OK.")
    assert eng._sigs == []                                                   # the code changed: the same command may now do something else


def test_different_reads_with_similar_output_are_not_a_loop(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    for i in range(12):
        eng._loop_guard("read_file", {"path": f"f{i}.py"}, "x = 1")
    assert not eng._abort_loop and not [m for m in eng.ctx.messages if m["role"] == "system"]


# ── gates are bounded ───────────────────────────────────────────────────────

def test_all_final_gates_together_send_the_model_back_at_most_four_times(tmp_path):
    from agent import plan as plan_mod

    llm_ = FakeLLM([call("write_file", path="app.py", content="VALUE = 1\n")] + [say("It works and is tested.")] * 8)
    eng, repo, out = engine(tmp_path, llm_, test_cmd="python3 check.py")
    (repo / "check.py").write_text("import app\nassert app.VALUE == 2\n")
    eng.plan.plan = plan_mod.Plan("t", [plan_mod.Item(1, "a"), plan_mod.Item(2, "b")])
    eng.config = {**CFG, "coding": {"claims_check": True}}
    res = eng.turn("make check.py pass")
    # changed on purpose (B4): open acceptance items no longer end silently - the answer goes out labelled NOT VERIFIED, stopped="gate"
    assert res.text.startswith("NOT VERIFIED: ") and res.text.endswith("It works and is tested.") and res.stopped == "gate" and res.verified is False
    # the first pure-text answer with open items is sent back once (announce rule, not a gate nudge); then at most four gate nudges
    assert len(llm_.calls) <= 1 + 1 + 1 + 4 and "accepting the answer" in out.getvalue()


def test_a_real_run_satisfies_the_claims_check_but_a_compile_check_does_not(tmp_path):
    llm_ = FakeLLM([call("write_file", path="m.py", content="print('hi')\n"), call("run", command="python3 m.py"), say("It works: prints hi.")])
    eng, repo, _ = engine(tmp_path, llm_)
    eng.config = {**CFG, "coding": {"claims_check": True}}
    assert eng.turn("write m.py").text == "It works: prints hi." and len(llm_.calls) == 3


def test_verification_never_waits_for_input_without_a_terminal(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    eng, repo, out = engine(tmp_path, FakeLLM([call("write_file", path="a.py", content="x = 1\n"), say("Done.")]), mode="medium",
                            test_cmd="python3 check.py")
    eng._explicit_approve = False
    eng.approve_fn = lambda d, r: pytest.fail("must not ask without a terminal")
    assert eng.turn("write a.py").text == "Done." and "skipped" in out.getvalue()


# ── running things ──────────────────────────────────────────────────────────

def test_background_children_die_with_the_command(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    t0 = time.time()
    code, out, _s = eng.tools._exec("(sleep 1; touch marker) & echo started", 30)     # the executor itself (the policy asks about `&`)
    assert code == 0 and "started" in out and time.time() - t0 < 1.0                       # a held pipe no longer blocks until the child exits
    time.sleep(1.4)
    assert not (repo / "marker").exists()


def test_timeout_kills_grandchildren_that_hold_the_output_pipe(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    t0 = time.time()
    code, out, _s = eng.tools._exec("bash -c 'sleep 30' & sleep 30", 1)
    assert code is None and "timed out after 1s" in out and time.time() - t0 < 4


def test_stdin_is_closed_and_output_is_capped_and_interleaved(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    assert "exit code 0" in eng.execute("run", {"command": "cat", "timeout": 5})              # no stdin: EOF at once, no hang
    out = eng.execute("run", {"command": "echo one; echo two >&2; echo three"})
    assert out.index("one") < out.index("two") < out.index("three")
    t0 = time.time()
    flood = eng.execute("run", {"command": "yes 0123456789abcdef", "timeout": 30})
    assert "exit code stopped" in flood and "more than 8 MB" in flood and len(flood) < 6000 and time.time() - t0 < 10


def test_a_game_with_lots_of_input_and_unicode_output(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "g.py").write_text("import sys\nfor line in sys.stdin:\n    print('» ' + line.strip())\n", encoding="utf-8")
    out = eng.execute("run", {"command": "python3 g.py", "stdin": "".join(f"cmd {i} grüße\n" for i in range(20000))})
    assert "exit code 0" in out and "» cmd 19999 grüße" in out


# ── writes ──────────────────────────────────────────────────────────────────

def test_an_interrupted_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "a.py").write_text("old = 1\n")
    monkeypatch.setattr(T.os, "replace", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        eng.tools.write_file("a.py", "new = 2\n" * 1000)
    assert (repo / "a.py").read_text() == "old = 1\n" and [p.name for p in repo.iterdir()] == ["a.py"]


def test_a_full_disk_is_a_clear_error_and_nothing_is_half_written(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "a.py").write_text("old = 1\n")
    real = os.fdopen

    def full(fd, *a, **k):
        f = real(fd, *a, **k)
        f.write = lambda *_: (_ for _ in ()).throw(OSError(28, "No space left on device"))
        return f

    monkeypatch.setattr(T.os, "fdopen", full)
    msg = eng.execute("write_file", {"path": "a.py", "content": "new = 2\n"})
    assert msg.startswith("Error running write_file") and "No space left" in msg
    assert (repo / "a.py").read_text() == "old = 1\n" and len(list(repo.iterdir())) == 1 and "a.py" not in eng.tools.dirty


def test_read_only_files_are_never_written_and_big_content_is_kept(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    save = repo / "world.json"
    save.write_text('{"room": 1}\n')
    save.chmod(0o444)
    try:
        big = json.dumps({"rooms": list(range(2000))})
        for name, args in [("write_file", {"path": "world.json", "content": big}), ("edit_file", {"path": "world.json", "search": "1", "replace": "2"}),
                           ("append_file", {"path": "world.json", "content": "x" * 400}), ("delete_file", {"path": "world.json"}),
                           ("apply_patch", {"patch": "@@\n-{\"room\": 1}\n+{\"room\": 2}\n", "path": "world.json"}),
                           ("move_file", {"src": "world.json", "dst": "w2.json"})]:
            msg = eng.execute(name, args)
            assert "read-only" in msg and "NOT changed" in msg, (name, msg)
        assert save.read_text() == '{"room": 1}\n' and stat.S_IMODE(save.stat().st_mode) == 0o444
        assert (repo / ".scratch" / "unsaved" / "world.json").read_text().strip() == big
    finally:
        save.chmod(0o644)


def test_writes_keep_crlf_mode_and_missing_trailing_newline(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    win = repo / "w.py"
    win.write_bytes(b"def f():\r\n    return 1\r\n\r\nx = f()\r\n")
    win.chmod(0o755)
    shown = eng.execute("read_file", {"path": "w.py"})
    assert "\r" not in shown and "CRLF" in shown
    assert eng.execute("edit_file", {"path": "w.py", "search": "def f():\n    return 1", "replace": "def f():\n    return 2"}).startswith("edited")
    assert win.read_bytes() == b"def f():\r\n    return 2\r\n\r\nx = f()\r\n" and stat.S_IMODE(win.stat().st_mode) == 0o755
    assert eng.execute("apply_patch", {"patch": "@@\n-x = f()\n+x = f() + 1\n", "path": "w.py"}).startswith("patched")
    assert win.read_bytes().endswith(b"x = f() + 1\r\n") and b"\n" not in win.read_bytes().replace(b"\r\n", b"")
    eng.execute("append_file", {"path": "w.py", "content": "y = 3\n" * 60})
    assert b"\n" not in win.read_bytes().replace(b"\r\n", b"")
    bare = repo / "n.txt"
    bare.write_text("a\nb")
    eng.execute("edit_file", {"path": "n.txt", "search": "a", "replace": "A"})
    assert bare.read_text() == "A\nb"


def test_binary_files_are_not_edited_and_an_overwrite_is_recoverable(tmp_path, monkeypatch):
    from agent import undo

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    blob = repo / "img.png"
    blob.write_bytes(b"\x89PNG\x00\x00\x01binary")
    for name, args in [("edit_file", {"path": "img.png", "search": "PNG", "replace": "GIF"}),
                       ("append_file", {"path": "img.png", "content": "x" * 400}),
                       ("apply_patch", {"patch": "@@\n-PNG\n+GIF\n", "path": "img.png"})]:
        assert "binary" in eng.execute(name, args), name
    assert blob.read_bytes() == b"\x89PNG\x00\x00\x01binary"
    trashed = []
    monkeypatch.setattr(undo, "trash", lambda p: trashed.append(p.read_bytes()) or p.unlink())
    msg = eng.execute("write_file", {"path": "img.png", "content": "<svg/>"})
    assert msg.startswith("wrote img.png") and "moved to the trash" in msg and trashed == [b"\x89PNG\x00\x00\x01binary"]


def test_unicode_paths_and_a_huge_single_line_file(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    assert eng.execute("write_file", {"path": "räume/über.py", "content": "x = 'ü'\n"}).startswith("wrote räume/über.py")
    assert "x = 'ü'" in eng.execute("read_file", {"path": "räume/über.py"})
    (repo / "min.js").write_text("var a=1;" * 40000)
    out = eng.execute("read_file", {"path": "min.js"})
    assert len(out) < 20000 and "1 lines" in out
    part = eng.execute("read_file", {"path": "min.js", "start": 1, "end": 1})
    assert len(part) < 15000 and "truncated" in part


# ── context and session ─────────────────────────────────────────────────────

def test_a_nearly_full_window_is_trimmed_even_when_compaction_fails(tmp_path):
    class Broken:
        def chat(self, *a, **k):
            raise httpx.ConnectError("down")

    ev = EventLog("t", tmp_path / "ev.jsonl")
    cm = C.ContextManager(Broken(), {"models": {"background": "m"}}, 10000, ev, tmp_path)
    cm.add({"role": "user", "content": "task"})
    for i in range(12):
        cm.add({"role": "assistant", "content": json.dumps({"tool": "run", "arguments": {"command": f"c{i}"}})})
        cm.add({"role": "tool", "content": "exit code 0\n" + "output line\n" * 250})
    cm.last_prompt = 9000
    notes = cm.maintain()
    assert any("nearly full" in n for n in notes) and [r for r in ev.recent if r["kind"] == "emergency_mask"]


def test_session_save_is_atomic_and_round_trips(tmp_path, monkeypatch):
    from agent.codeagent import session as S

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.start()
    eng.ctx.add_user("hello")
    S.save_session(eng)
    from agent.codeagent import sessionstore as SS

    f = SS.folder_dir(eng.root) / f"{eng.session_id}.json"
    assert json.loads(f.read_text())["requests"] == ["hello"]
    assert sorted(p.name for p in f.parent.iterdir()) == sorted([f.name, f"{eng.session_id}.meta.json"])     # no temp files left
    eng2, _r, _o = engine(tmp_path, FakeLLM([]))
    assert S.resume_into(eng2, SS.latest(eng.root)) and eng2.ctx.user_requests == ["hello"]


def test_long_or_slow_run_output_is_kept_whole_in_a_file(tmp_path):
    from agent.codeagent.tools import CodeTools  # noqa: F401  (engine fixture builds the tools)

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    res = eng.tools.run(command="python3 -c \"print('\\n'.join(f'line {i} ' + 'x' * 40 for i in range(400)))\"")
    assert "full output: .scratch/spill/run-1.txt (400 lines)" in res and "line 399" in (repo / ".scratch" / "spill" / "run-1.txt").read_text()
    short = eng.tools.run(command="echo hi")
    assert "full output" not in short


# ── second pass: slow runs, paging through output, failed edits ────────────

@pytest.mark.parametrize("command, stem", [
    ("cd .scratch && python3 t.py 2>&1 | sed -n '1,45p'", "cd .scratch && python3 t.py 2>&1 "),
    ("printf '1\\nquit\\n' | python3 main.py 2>&1 | grep -n 'a\\|b' | head -40", "printf '1\\nquit\\n' | python3 main.py 2>&1 "),
    ("python3 main.py |& tail -5", "python3 main.py "),
    ("python3 a.py | python3 b.py", None),                  # the last stage is a program, not a filter
    ("python3 a.py || head x", None),                       # `||` is not a pipe
    ("echo 'a | sed x'", None),                             # quoted
    ("python3 $(echo a.py) | head", None),                  # substitutions are left alone
    ("python3 a.py | head; echo done", None),               # only the last pipeline counts
    ("python3 a.py 2>&1 | grep x 2>&1 | head", "python3 a.py 2>&1 "),
    ("python3 a.py & python3 b.py | tail", "python3 a.py & python3 b.py "),
])
def test_filter_split_finds_the_trailing_filters(command, stem):
    cut = T.filter_split(command)
    assert (command[:cut] if cut is not None else None) == stem


def test_a_filtered_run_keeps_the_unfiltered_output(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    res = eng.tools.run(command="python3 -c \"[print(f'line {i}') for i in range(300)]\" 2>&1 | sed -n '1,5p'")
    assert res.startswith("exit code 0") and "line 4" in res and "line 5" not in res
    assert "full output: .scratch/spill/run-1.txt (300 lines, the program's output before `| sed -n '1,5p'`)" in res
    assert "line 299" in (repo / ".scratch" / "spill" / "run-1.txt").read_text()
    assert eng.tools.run(command="false | cat").startswith("exit code 0")        # the pipeline's exit code is unchanged
    assert eng.tools.run(command="echo x | grep -q y").startswith("exit code 1")
    t0 = time.time()
    assert "y" in eng.tools.run(command="yes | head -1", timeout=20) and time.time() - t0 < 5    # an endless producer still stops
    flood = eng.tools.run(command="yes 0123456789abcdef | grep -c nothing", timeout=30)
    assert "more than 8 MB" in flood and time.time() - t0 < 15                                # the side copy counts against the cap
    assert not list(repo.glob("**/agent-run-*"))


def test_no_sleep_skips_time_sleep_and_a_sleepy_run_says_so(tmp_path, monkeypatch):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "slow.py").write_text("import time\nfor i in range(3):\n    time.sleep(10)\n    print('typed', i)\n")
    t0 = time.time()
    fast = eng.execute("run", {"command": "python3 slow.py", "no_sleep": "true"})
    assert fast.startswith("exit code 0") and "typed 2" in fast and time.time() - t0 < 5 and "no_sleep" not in fast
    monkeypatch.setattr(T, "SLOW_RUN_S", 0.5)
    (repo / "nap.py").write_text("import time\ntime.sleep(0.8)\nprint('done')\n")
    sleepy = eng.tools.run(command="python3 nap.py")
    assert "used only" in sleepy and "no_sleep=true" in sleepy
    assert "used only" not in eng.tools.run(command="python3 nap.py", no_sleep=True)
    assert "used only" not in eng.tools.run(command="sleep 0.8")                       # not Python: the switch would not help


def test_paging_through_the_same_output_gets_one_hint_and_a_reasoning_step(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 40
    (repo / ".scratch" / "spill").mkdir(parents=True)
    (repo / ".scratch" / "spill" / "run-1.txt").write_text("x\n")
    eng.tools.spill.last["run"] = ".scratch/spill/run-1.txt"
    lines = [f"room {i}: the harbour waits" for i in range(80)]
    for n in (15, 20):
        _play(eng, f"cd .scratch && python3 test_scripted.py 2>&1 | sed -n '1,{n}p'", "\n".join(lines[:n]), 17.6)
    assert not [m for m in eng.ctx.messages if m["role"] == "system"]
    _play(eng, "cd .scratch && python3 test_scripted.py 2>&1 | sed -n '1,25p'", "\n".join(lines[:25]), 17.6)
    hints = [m["content"] for m in eng.ctx.messages if m["role"] == "system"]
    assert len(hints) == 1 and "3 times" in hints[0] and "17.6s" in hints[0] and ".scratch/spill/run-1.txt" in hints[0]
    assert eng.think_steps == 1 and not eng._abort_loop


def test_different_programs_with_filters_are_not_paging(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.ctx.step = 40
    for i in range(5):
        _play(eng, f"python3 check_{i}.py | tail -3", f"ok {i}", 1.0)
    assert not [m for m in eng.ctx.messages if m["role"] == "system"]


def test_a_failed_edit_shows_the_real_lines_and_the_first_difference(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "t.py").write_text("import io\n\n\ndef run_script(inputs):\n    buf = io.StringIO()\n    return buf.getvalue()\n")
    res = eng.execute("edit_file", {"path": "t.py", "search": 'def run_script(inputs):\n    """Run the game."""\n    buf = io.StringIO()',
                                    "replace": "def run_script(inputs):\n    pass"})
    assert res.startswith("edit_file failed") and "starts at line 4" in res
    assert "differs from the file's line 5: 'buf = io.StringIO()'" in res and "    return buf.getvalue()" in res


def test_not_run_advice_matches_the_reason(tmp_path, monkeypatch):
    import sys as _sys

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: False, raising=False)
    eng, repo, _ = engine(tmp_path, FakeLLM([]), mode="auto")
    eng._explicit_approve = False
    res = eng.execute("run", {"command": "python3 main.py | tail -70 && rm -rf .scratch && mkdir -p .scratch"})
    assert res.startswith("NOT RUN") and "comes first" in res and "relative paths" not in res


def test_history_placeholder_is_never_written_into_a_file(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "d.py").write_text("X = 1\n")
    res = eng.tools.append_file(path="d.py", content="[1 lines, 33 chars - now on disk]")
    assert res.startswith("error") and "placeholder" in res and (repo / "d.py").read_text() == "X = 1\n"
    res = eng.tools.write_file(path="n.py", content="[12 lines, 300 chars - now on disk]", overwrite=True)
    assert res.startswith("error") and not (repo / "n.py").exists()
    res = eng.tools.append_file(path="d.py", content="Y = 2\n[1 lines, 33 chars - now on disk]\nZ = 3")      # real text around it: refused, never stripped
    assert res.startswith("error") and "placeholder" in res and (repo / "d.py").read_text() == "X = 1\n"


def test_the_history_shortening_marker_is_never_written_into_a_file(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "n.py").write_text('A = 1\nTEXT = "hello"\nB = 2\n')
    eng.execute("read_file", {"path": "n.py"})                           # replace_lines needs line numbers the model has really seen
    res = eng.execute("replace_lines", {"path": "n.py", "start": 2, "end": 2, "new_text": 'TEXT = "hello wor...[cut]'})
    assert "harness marker" in res and (repo / "n.py").read_text() == 'A = 1\nTEXT = "hello"\nB = 2\n'
    res = eng.execute("write_file", {"path": "m.py", "content": 'X = "a...[cut]"\n', "overwrite": True})
    assert "harness marker" in res and not (repo / "m.py").exists()
    (repo / "k.py").write_text('M = "...[cut]"\n')                     # a file that already uses the text legitimately may keep it
    eng.execute("read_file", {"path": "k.py"})
    assert "harness marker" not in eng.execute("replace_lines", {"path": "k.py", "start": 1, "end": 1, "new_text": 'M = "...[cut]"  # same'})


def test_a_placeholder_sent_to_replace_lines_or_insert_after_is_refused_with_an_explanation(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "n.py").write_text("A = 1\nB = 2\n")
    eng.execute("read_file", {"path": "n.py"})
    res = eng.execute("replace_lines", {"path": "n.py", "start": 1, "end": 1, "new_text": "[3 lines, 40 chars - now on disk]"})
    assert "placeholder" in res and (repo / "n.py").read_text() == "A = 1\nB = 2\n"
    res = eng.execute("insert_after", {"path": "n.py", "line": 1, "new_text": "[3 lines, 40 chars - now on disk]"})
    assert "placeholder" in res and (repo / "n.py").read_text() == "A = 1\nB = 2\n"
