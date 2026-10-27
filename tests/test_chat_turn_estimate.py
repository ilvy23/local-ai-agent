from agent.chat.turn import prompt_estimate


def test_prompt_estimate_counts_messages_and_a_dense_tool_schema():
    msgs = [{"role": "system", "content": "x" * 400}, {"role": "user", "content": "y" * 40}]
    assert prompt_estimate(msgs) == 110
    assert prompt_estimate(msgs, [{"name": "t" * 120}]) > 110
    assert prompt_estimate([{"role": "user", "content": None}]) == 0
