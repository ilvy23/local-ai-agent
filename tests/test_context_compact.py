from agent.context.compact import fold_messages, should_compact


def _msgs(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n)]


def test_no_fold_when_short():
    msgs = _msgs(2)
    result = fold_messages(msgs, keep_recent=3)
    assert not result.changed
    assert result.messages == msgs


def test_folds_old_turns():
    msgs = _msgs(10)
    result = fold_messages(msgs, keep_recent=3)
    assert result.changed
    assert result.folded == 7
    assert result.messages[0]["role"] == "system"
    assert "summary" in result.messages[0]["content"]
    assert len(result.messages) == 4


def test_preserves_leading_system():
    msgs = [{"role": "system", "content": "sys"}, *_msgs(10)]
    result = fold_messages(msgs, keep_recent=2)
    assert result.messages[0]["content"] == "sys"
    assert result.messages[1]["content"].startswith("[summary")


def test_custom_summarizer():
    msgs = _msgs(10)
    result = fold_messages(msgs, keep_recent=2, summarize=lambda m: f"folded {len(m)}")
    assert "folded 8" in result.messages[0]["content"]


def test_should_compact():
    assert should_compact(80, 100, 0.75)
    assert not should_compact(50, 100, 0.75)
    assert not should_compact(80, 0)
