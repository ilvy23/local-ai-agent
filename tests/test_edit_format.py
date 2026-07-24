from agent.coding.edit.format import EditKind, parse_edits


def test_whole_file_block():
    resp = "<<<< FILE: foo.py\nprint('hi')\n>>>>"
    result = parse_edits(resp)
    assert result.ok
    edit = result.edits[0]
    assert edit.kind is EditKind.WHOLE_FILE
    assert edit.path == "foo.py"
    assert edit.replace == "print('hi')"


def test_whole_file_preserves_internal_blank_lines():
    resp = "<<<< FILE: a.py\nx = 1\n\ny = 2\n>>>>"
    edit = parse_edits(resp).edits[0]
    assert edit.replace == "x = 1\n\ny = 2"


def test_search_replace_block():
    resp = (
        "<<<< FILE: bar.py\n"
        "------ SEARCH\n"
        "old = 1\n"
        "====== REPLACE\n"
        "old = 2\n"
        ">>>>"
    )
    edit = parse_edits(resp).edits[0]
    assert edit.kind is EditKind.SEARCH_REPLACE
    assert edit.search == "old = 1"
    assert edit.replace == "old = 2"


def test_multiple_blocks():
    resp = (
        "<<<< FILE: a.py\na = 1\n>>>>\n"
        "<<<< FILE: b.py\nb = 2\n>>>>"
    )
    result = parse_edits(resp)
    assert len(result.edits) == 2


def test_no_blocks_is_malformed():
    result = parse_edits("I think you should change line 4.")
    assert not result.ok
    assert result.malformed


def test_unclosed_block_is_malformed():
    result = parse_edits("<<<< FILE: a.py\nx = 1\n")
    assert not result.ok
    assert "never closed" in result.malformed[0]


def test_half_search_replace_is_malformed():
    resp = "<<<< FILE: a.py\n------ SEARCH\nfoo\n>>>>"
    result = parse_edits(resp)
    assert not result.ok
    assert "REPLACE" in result.malformed[0]


def test_replace_before_search_is_malformed():
    resp = (
        "<<<< FILE: a.py\n"
        "====== REPLACE\n"
        "new\n"
        "------ SEARCH\n"
        "old\n"
        ">>>>"
    )
    result = parse_edits(resp)
    assert not result.ok


def test_empty_search_is_malformed():
    resp = "<<<< FILE: a.py\n------ SEARCH\n====== REPLACE\nnew\n>>>>"
    result = parse_edits(resp)
    assert not result.ok


def test_mixed_valid_and_malformed():
    resp = (
        "<<<< FILE: a.py\na = 1\n>>>>\n"
        "<<<< FILE: b.py\n------ SEARCH\nx\n>>>>"
    )
    result = parse_edits(resp)
    assert len(result.edits) == 1
    assert len(result.malformed) == 1
    assert not result.ok
