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


def test_extracts_json_with_raw_newline_in_string():
    # A model writing real multi-line file content into this fallback text
    # format often emits a literal newline instead of an escaped \n, which
    # JSON forbids inside a string - this used to silently kill the whole
    # tool call, effectively forcing single-line-only content through.
    text = '{"tool": "write_file", "arguments": {"path": "x.py", "content": "line1\nline2\nline3"}}'
    assert extract_json_value(text, "{", "}") == {
        "tool": "write_file",
        "arguments": {"path": "x.py", "content": "line1\nline2\nline3"},
    }


def test_extracts_json_with_raw_tab_and_carriage_return_in_string():
    text = '{"a": "col1\tcol2\r\nrow2"}'
    assert extract_json_value(text, "{", "}") == {"a": "col1\tcol2\r\nrow2"}


def test_extracts_json_with_both_invalid_escape_and_raw_newline():
    # A candidate can have more than one defect at once; the repair loop
    # must keep alternating repairs rather than giving up after the first.
    text = '{"tool": "current\\_time", "note": "line1\nline2"}'
    assert extract_json_value(text, "{", "}") == {
        "tool": "current_time",
        "note": "line1\nline2",
    }


def test_control_character_repair_does_not_touch_structural_whitespace():
    # Ordinary formatting newlines between JSON tokens (outside any string)
    # are already valid JSON and must be left exactly as they are.
    text = '{\n  "tool": "current_time",\n  "arguments": {}\n}'
    assert extract_json_value(text, "{", "}") == {"tool": "current_time", "arguments": {}}
