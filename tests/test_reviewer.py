from agent.subagents.roles.reviewer import REVIEWER_SPEC, review_diff
from agent.subagents.spawn import spawn


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, messages, model, stream=False, num_ctx=None, temperature=None, options=None):
        self.calls.append((model, messages))
        return iter([self.reply])


def test_reviewer_returns_concerns():
    client = FakeClient("- off-by-one in loop\n- missing test for empty input")
    review = review_diff(client, diff="- x\n+ y", task="fix loop", model="m")
    assert len(review.concerns) == 2
    assert not review.clean


def test_reviewer_clean_on_none():
    client = FakeClient("NONE")
    review = review_diff(client, diff="- x\n+ y", task="fix", model="m")
    assert review.clean


def test_reviewer_cannot_approve_language_in_system():
    assert "cannot approve" in REVIEWER_SPEC.system


def test_spawn_truncates_long_output():
    client = FakeClient("x" * 5000)
    result = spawn(client, REVIEWER_SPEC, {"task": "t", "diff": "d"}, model="m")
    assert result.truncated


def test_spawn_uses_given_model():
    client = FakeClient("NONE")
    spawn(client, REVIEWER_SPEC, {"task": "t", "diff": "d"}, model="big")
    assert client.calls[0][0] == "big"


def test_reviewer_parses_numbered():
    client = FakeClient("1. first concern\n2. second concern")
    review = review_diff(client, diff="d", task="t", model="m")
    assert len(review.concerns) == 2
