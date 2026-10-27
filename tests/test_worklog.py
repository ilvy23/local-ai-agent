from agent.memory import worklog


class FakeLLM:
    def __init__(self, reply="OPEN: x\nDONE: y"):
        self.reply, self.calls = reply, []

    def chat(self, messages, model, stream=True, **kw):
        self.calls.append((messages, kw))
        return iter([self.reply])


CFG = {"models": {"background": "bg"}}


def _loop(n):
    msgs = [{"role": "system", "content": "persona"}, {"role": "user", "content": "build the thing"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f'{{"tool": "write_file", "arguments": {{"path": "f{i}.py"}}}}'},
                 {"role": "tool", "content": f"wrote f{i}.py " + "x" * 50}]
    return msgs


def test_should_compact_threshold(monkeypatch):
    monkeypatch.delenv("AGENT_COMPACT_AT", raising=False)
    assert not worklog.should_compact(40000, 65000)
    assert worklog.should_compact(46000, 65000)
    assert not worklog.should_compact(0, 65000) and not worklog.should_compact(9999, None)
    monkeypatch.setenv("AGENT_COMPACT_AT", "0.5")
    assert worklog.should_compact(33000, 65000)


def test_compact_keeps_task_and_tail_and_inserts_log():
    msgs = _loop(10)
    llm = FakeLLM()
    out = worklog.compact_turn_messages(llm, CFG, msgs)
    assert out[:2] == msgs[:2]                                  # persona + task verbatim
    assert out[2]["role"] == "system" and out[2]["content"].startswith(worklog.WORKLOG_MARKER)
    assert "OPEN: x" in out[2]["content"]
    assert out[3:] == msgs[-6:] and out[3]["role"] == "assistant"   # tail starts at a call
    assert len(out) < len(msgs)
    sent, kw = llm.calls[0]
    assert kw.get("background") is True and "f0.py" in sent[1]["content"]


def test_compact_skips_small_loops_and_survives_model_failure():
    assert worklog.compact_turn_messages(FakeLLM(), CFG, _loop(2)) is None

    class Boom(FakeLLM):
        def chat(self, *a, **k):
            raise RuntimeError("down")

    assert worklog.compact_turn_messages(Boom(), CFG, _loop(10)) is None


def test_second_compaction_merges_the_earlier_log():
    first = worklog.compact_turn_messages(FakeLLM("GOAL: g\nOPEN: a"), CFG, _loop(10))
    grown = first + _loop(10)[2:]
    llm = FakeLLM("GOAL: g\nOPEN: b")
    second = worklog.compact_turn_messages(llm, CFG, grown)
    assert sum(1 for m in second if m["content"].startswith(worklog.WORKLOG_MARKER)) == 1
    assert "EARLIER LOG" in llm.calls[0][0][1]["content"] and "OPEN: a" in llm.calls[0][0][1]["content"]
