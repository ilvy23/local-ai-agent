from agent.context.truncate import truncate_output


def test_short_output_untouched():
    out = "line1\nline2\nline3"
    result = truncate_output(out, head=40, tail=40)
    assert not result.was_truncated
    assert result.text == out


def test_long_output_truncated():
    out = "\n".join(str(i) for i in range(200))
    result = truncate_output(out, head=5, tail=5)
    assert result.was_truncated
    assert "truncated" in result.text
    assert result.text.startswith("0\n1")
    assert result.text.rstrip().endswith("199")


def test_spillover_writes_full_output(tmp_path):
    out = "\n".join(str(i) for i in range(200))
    result = truncate_output(out, spill_dir=tmp_path, head=5, tail=5, ref_id="abc")
    assert result.spill_path.exists()
    assert result.spill_path.read_text() == out
    assert "abc.txt" in result.text


def test_original_line_count():
    out = "\n".join(str(i) for i in range(200))
    result = truncate_output(out, head=5, tail=5)
    assert result.original_lines == 200
