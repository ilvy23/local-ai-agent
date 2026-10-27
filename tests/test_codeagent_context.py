from __future__ import annotations

import json

from agent.codeagent import context as C
from agent.codeagent.eventlog import EventLog


class Summarizer:
    def __init__(self, text="DONE: wrote main.py\nSTATE: ok\nDECISIONS: none\nOPEN: tests\nPITFALLS: none"):
        self.text, self.calls = text, []

    def chat(self, messages, model=None, stream=False, background=False, **kw):
        self.calls.append(messages)
        yield self.text


def cm(tmp_path, ctx=10000, llm=None):
    return C.ContextManager(llm or Summarizer(), {"models": {"background": "m"}}, ctx, EventLog("t", tmp_path / "e.jsonl"), tmp_path)


def fill(manager, n, size=3000):
    for i in range(n):
        manager.add({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": f"f{i}.py"}})})
        manager.add({"role": "tool", "content": f"f{i}.py ({size} chars)\n" + "x = 1\n" * (size // 6)})


def test_compact_call_drops_payloads_but_keeps_the_shape_the_model_imitates():
    big = "print('x')\n" * 500
    c = C.compact_call("write_file", {"path": "a.py", "content": big})
    assert c["arguments"]["path"] == "a.py" and "now on disk" in c["arguments"]["content"] and len(json.dumps(c)) < 200
    # the model copies its own history: `edits` must stay a LIST of {search, replace} objects, not become a status string
    e = C.compact_call("edit_file", {"path": "a.py", "edits": [{"search": big, "replace": big}, {"search": "a", "replace": "b"}]})
    edits = e["arguments"]["edits"]
    assert isinstance(edits, list) and len(edits) == 2 and set(edits[0]) == {"search", "replace"}
    assert len(edits[0]["search"]) < 200 and edits[0]["search"].endswith("...[cut]") and edits[1] == {"search": "a", "replace": "b"}
    flat = C.compact_call("edit_file", {"path": "a.py", "search": big, "replace": "x"})
    assert flat["arguments"]["replace"] == "x" and len(flat["arguments"]["search"]) < 200 and "edits" not in flat["arguments"]
    diff = "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n-old\n+new\n" + "+filler\n" * 400
    pc = C.compact_call("apply_patch", {"patch": diff})["arguments"]["patch"]
    assert pc.startswith("--- a/m.py\n+++ b/m.py\n@@") and len(pc) < 600                 # still a real diff header, not a status text
    assert C.compact_call("run", {"command": "ls"}) == {"tool": "run", "arguments": {"command": "ls"}}


def test_masking_happens_once_in_bulk_then_rearms(tmp_path):
    m = cm(tmp_path, ctx=10000)
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 14)
    m.last_prompt, m._since = 7500, 0                       # 75% full -> over the 72% mask line
    cleared = []
    m.on_clear_cache = lambda: cleared.append(1)
    notes = m.maintain()
    assert any("trimmed" in n for n in notes) and cleared
    old_tools = [x for x in m.messages if x["role"] == "tool"]
    assert sum("omitted" in x["content"] for x in old_tools) >= 8 and "x = 1" in old_tools[-1]["content"]   # recent ones intact
    before = [dict(x) for x in m.messages]
    m.last_prompt, m._since = 7300, 0                       # still above the line but below the re-arm level
    m.maintain()
    assert m.messages == before                             # no second rewrite: the prefix (and the server cache) stays stable
    assert [r for r in m.events.recent if r["kind"] == "mask"]


def test_compaction_keeps_ledger_requests_and_the_head(tmp_path):
    (tmp_path / "main.py").write_text("def main():\n    pass\n")
    (tmp_path / "broken.py").write_text("def f(:\n")
    llm = Summarizer()
    m = cm(tmp_path, ctx=10000, llm=llm)
    m.add({"role": "system", "content": "sys"})
    m.add_user("build the game in ~/Projects, plan first, six locations")
    m.ledger.note_file("main.py", "def main():\n    pass\n", 3, "wrote")
    m.ledger.note_file("broken.py", "def f(:\n", 4, "wrote")
    m.ledger.note_file("gone.py", "x=1\n", 5, "wrote")
    m.ledger.note_run(6, "python3 main.py", 0, "ok")
    m.ledger.note_run(7, "pytest -q", 1, "1 failed")
    fill(m, 12)
    m.last_prompt, m._since = 7200, 0                       # 72% -> compaction
    note = m.compact()
    assert note and "compacted" in note
    log = next(x for x in m.messages if x["role"] == "system" and x["content"].startswith("WORK LOG"))["content"]
    assert "main.py: 2 lines" in log and "syntax ok" in log
    assert "broken.py" in log and "SYNTAX ERROR" in log and "gone.py: MISSING" in log
    assert "`pytest -q` -> exit 1" in log
    assert "plan first, six locations" not in log and "plan first, six locations" in m.messages[1]["content"]   # in the head: no 2nd copy
    sent = llm.calls[0][1]["content"]
    assert "USER REQUESTS" in sent and "plan first, six locations" in sent
    assert m.messages[0]["content"] == "sys" and m.messages[1]["role"] == "user"       # head intact
    assert len(m.messages) < 12 and m.messages[-1]["role"] == "tool"
    assert [r for r in m.events.recent if r["kind"] == "compact"][0]["before"] >= 7200


def test_a_second_compaction_merges_the_earlier_log(tmp_path):
    llm = Summarizer()
    m = cm(tmp_path, llm=llm)
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    fill(m, 10)
    assert m.compact()
    fill(m, 10)
    assert m.compact()
    assert "EARLIER LOG" in llm.calls[1][1]["content"]
    assert sum(1 for x in m.messages if x["content"].startswith("WORK LOG")) == 1


def test_fold_history_at_the_start_of_a_prompt(tmp_path):
    m = cm(tmp_path, ctx=10000)
    m.add({"role": "system", "content": "sys"})
    m.add_user("first task")
    fill(m, 12)
    m.add({"role": "assistant", "content": "done with the first task"})
    m.last_prompt, m._since = 7500, 0
    m.add_user("second task, please continue")
    assert m.fold_history() is not None
    assert m.messages[0]["role"] == "system" and m.messages[-1] == {"role": "user", "content": "second task, please continue"}
    assert any(x["content"].startswith("WORK LOG") for x in m.messages)
    assert "first task" in next(x for x in m.messages if x["content"].startswith("WORK LOG"))["content"]       # asked verbatim
    small = cm(tmp_path)
    small.add({"role": "system", "content": "sys"}); small.add_user("x")
    assert small.fold_history() is None


def test_masking_keeps_the_outline_of_files_the_model_read(tmp_path):
    (tmp_path / "story.py").write_text("class Story:\n    def start(self):\n        pass\n\n    def ending(self, kind):\n        return kind\n\n" + "# filler\n" * 400)
    m = cm(tmp_path, ctx=10000)
    m.add({"role": "system", "content": "sys"})
    m.add_user("task")
    m.add({"role": "assistant", "content": json.dumps({"tool": "read_file", "arguments": {"path": "story.py", "start": 1, "end": 400}})})
    m.add({"role": "tool", "content": "story.py (407 lines)\n" + "x = 1\n" * 300})
    m.add({"role": "assistant", "content": json.dumps({"tool": "search", "arguments": {"pattern": "ending"}})})
    m.add({"role": "tool", "content": "\n".join(f"story.py:{i}: ending" for i in range(60))})
    m.add({"role": "assistant", "content": json.dumps({"tool": "run", "arguments": {"command": "pytest"}})})
    m.add({"role": "tool", "content": "\n".join(f"line {i} of output" for i in range(80))})
    fill(m, 8)
    masked, saved = C.mask_observations(m.messages, tmp_path, keep_last=4)
    read_stub = next(x for x in masked if x["role"] == "tool" and "old read of story.py" in x["content"])["content"]
    assert "class Story" in read_stub and "def ending(self, kind)" in read_stub and "Read only the exact line range" in read_stub
    assert len(read_stub) < 600 and saved > 2000                                  # the model still knows the structure at a fraction of the size
    assert any("[old search results trimmed]" in x["content"] and "more lines omitted" in x["content"] for x in masked if x["role"] == "tool")
    run_stub = next(x["content"] for x in masked if x["role"] == "tool" and "old command output" in x["content"])
    assert "line 0 of output" in run_stub and "line 79 of output" in run_stub     # head and tail of a command survive
    again, saved2 = C.mask_observations(masked, tmp_path, keep_last=4)
    assert saved2 == 0 and again == masked                                          # already-masked results are left alone


def test_large_files_return_the_first_lines_with_a_continue_line_and_reads_are_counted(tmp_path):
    from agent.codeagent import profile as PR
    from agent.codeagent.tools import CodeTools

    big = "class World:\n    def build(self):\n        pass\n\n" + "# padding line\n" * 300
    (tmp_path / "world.py").write_text(big)
    ctx = cm(tmp_path)
    tools = CodeTools(tmp_path, PR.detect(tmp_path), ctx)
    out = tools.read_file("world.py")
    assert "class World" in out and "--- lines 1-250 ---" in out and out.endswith("lines 1-250 of 304; continue with start=251")   # was: an outline only
    assert tools.read_lines == 250
    ol = tools.read_file("world.py", outline=True)
    assert "OUTLINE" in ol and "class World" in ol and tools.read_lines == 250        # an outline is not a read of the content
    part = tools.read_file("world.py", start=1, end=40)
    assert "class World" in part and tools.read_lines == 290


def test_recent_writes_keep_their_real_text_old_ones_are_masked(tmp_path):
    import json as _json
    from agent.codeagent.context import PLACEHOLDER, mask_observations

    big = "x = 1\n" * 400
    msgs = []
    for i in range(14):
        msgs.append({"role": "assistant", "content": _json.dumps({"tool": "write_file", "arguments": {"path": f"f{i}.py", "content": big}})})
        msgs.append({"role": "tool", "content": f"wrote f{i}.py"})
    out, saved = mask_observations(msgs, tmp_path, keep_last=10)
    assert saved > 0
    first, last = _json.loads(out[0]["content"]), _json.loads(out[-2]["content"])
    assert PLACEHOLDER.search(first["arguments"]["content"]) and first["arguments"]["path"] == "f0.py"
    assert last["arguments"]["content"] == big


def test_replace_lines_text_stays_clear_while_recent_and_becomes_the_placeholder_when_old_never_a_cut_marker(tmp_path):
    import json as _json
    from agent.codeagent.context import PLACEHOLDER, compact_call, mask_observations

    big = "line of new text\n" * 120
    assert "[cut]" not in _json.dumps(compact_call("replace_lines", {"path": "a.py", "start": 1, "end": 2, "new_text": big}))
    assert PLACEHOLDER.search(compact_call("insert_after", {"path": "a.py", "line": 3, "new_text": big})["arguments"]["new_text"])
    msgs = []
    for i in range(14):
        msgs.append({"role": "assistant", "content": _json.dumps({"tool": "replace_lines", "arguments": {"path": f"f{i}.py", "start": 1, "end": 1, "new_text": big}})})
        msgs.append({"role": "tool", "content": "edited"})
    out, saved = mask_observations(msgs, tmp_path, keep_last=10)
    assert saved > 0
    assert PLACEHOLDER.search(_json.loads(out[0]["content"])["arguments"]["new_text"]) and _json.loads(out[-2]["content"])["arguments"]["new_text"] == big
