from agent.jsonx import extract_json_value


def test_extracts_clean_json_object():
    assert extract_json_value('{"a": 1}', "{", "}") == {"a": 1}


def test_extracts_clean_json_array():
    assert extract_json_value('["a", "b"]', "[", "]") == ["a", "b"]


def test_extracts_json_object_fenced_in_code_block():
    text = '```json\n{"tool": "current_time", "arguments": {}}\n```'
    assert extract_json_value(text, "{", "}") == {"tool": "current_time", "arguments": {}}


def test_extracts_json_object_wrapped_in_junk_text():
    text = 'Sure, let me help.\n{"tool": "current_time", "arguments": {}}\nDone.'
    assert extract_json_value(text, "{", "}") == {"tool": "current_time", "arguments": {}}


def test_returns_none_when_no_json_present():
    assert extract_json_value("I could not find any facts, sorry.", "{", "}") is None


def test_skips_unparseable_bracket_pair_and_finds_next():
    text = '{not valid} then {"tool": "current_time", "arguments": {}}'
    assert extract_json_value(text, "{", "}") == {"tool": "current_time", "arguments": {}}


def test_extract_json_value_returns_none_for_wrong_type():
    # Requesting an object but the only bracketed content is an array-shaped string
    assert extract_json_value('["a", "b"]', "{", "}") is None


def test_extracts_json_with_invalid_backslash_escape():
    # Some models (e.g. mixtral) emit stray backslashes before underscores,
    # which is not a valid JSON escape and would otherwise make json.loads
    # reject an object that is otherwise well-formed.
    text = '{"tool": "current\\_time", "arguments": {}}'
    assert extract_json_value(text, "{", "}") == {"tool": "current_time", "arguments": {}}


def test_extracts_json_with_multiple_invalid_backslash_escapes():
    text = '{"tool": "current\\_time", "note": "a\\_b\\_c"}'
    assert extract_json_value(text, "{", "}") == {
        "tool": "current_time",
        "note": "a_b_c",
    }


def test_does_not_repair_unrelated_parse_error_even_with_invalid_escape():
    # An unquoted bare literal is a structural defect unrelated to escaping;
    # the presence of an invalid escape elsewhere in the same object must not
    # trigger a blanket repair that could silently mangle content of a
    # candidate that was broken for a completely different reason.
    text = '{"flag": tru, "tool": "current\\_time"}'
    assert extract_json_value(text, "{", "}") is None


def test_escape_repair_loop_terminates_on_pathological_input():
    # Many invalid escapes in a row must not hang or infinite-loop.
    text = '{"a": "' + "\\_" * 100 + '"}'
    result = extract_json_value(text, "{", "}")
    assert result is None or result == {"a": "_" * 100}
