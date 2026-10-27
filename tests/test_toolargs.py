

def test_stringified_lists_with_code_inside_are_repaired():
    from agent.toolargs import _coerce, validate
    from agent.tools.registry import Tool

    raw_newlines = '[{"search": "def f():\n    return 1", "replace": "def f():\n    return 2"}]'      # invalid strict JSON (raw newlines)
    value, ok = _coerce(raw_newlines, "array")
    assert ok and value[0]["replace"].endswith("return 2")
    py_literal = "[{'search': 'a = 1', 'replace': 'a = 2'}]"
    assert _coerce(py_literal, "array") == ([{"search": "a = 1", "replace": "a = 2"}], True)
    assert _coerce('Here: [{"search": "x", "replace": "y"}] done', "array")[1] is True
    assert _coerce("not a list at all", "array")[1] is False and _coerce('{"a": 1}', "array")[1] is False

    tool = Tool("edit_file", "d", {"type": "object", "properties": {
        "path": {"type": "string"}, "edits": {"type": "array", "items": {"type": "object", "properties": {"search": {"type": "string"}, "replace": {"type": "string"}}}}},
        "required": ["path", "edits"]}, lambda **kw: "ok")
    fixed, err = validate(tool, {"path": "a.py", "edits": raw_newlines})
    assert err is None and isinstance(fixed["edits"], list)
    _fixed, err = validate(tool, {"path": "a.py", "edits": "unparseable <<<"})
    assert "must be array" in err and 'Example: "edits": [{"search": "...", "replace": "..."}]' in err and "not a quoted string" in err
