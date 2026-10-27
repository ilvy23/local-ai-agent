import io

from rich.console import Console

from agent import liveview, tui


def _console():
    return Console(file=io.StringIO(), force_terminal=False, width=100, color_system=None)


def test_normalize_arguments_unwraps_json_strings_and_wrappers():
    n = tui._normalize_arguments
    assert n({"path": "a", "content": "b"}) == {"path": "a", "content": "b"}
    assert n('{"path": "a"}') == {"path": "a"}
    assert n({"arguments": '{"path": "a", "content": "b"}'}) == {"path": "a", "content": "b"}
    assert n({"arguments": {"path": "a"}}) == {"path": "a"}
    assert n("not json") == {} and n(None) == {}
    call = tui._extract_tool_call({"tool_calls": [{"function": {"name": "write_file",
                                   "arguments": {"arguments": '{"path": "/x/y.py", "content": "1"}'}}}]})
    assert call == {"name": "write_file", "arguments": {"path": "/x/y.py", "content": "1"}}


def test_ask_more_steps_defaults_to_no_off_terminal():
    assert tui._ask_more_steps(_console(), 60) is False
