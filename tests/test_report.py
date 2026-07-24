from agent.coding.verify.report import enrich_traceback, summarize_failure


def test_enrich_splices_source(tmp_path):
    mod = tmp_path / "mod.py"
    mod.write_text("def a():\n    return None\n\ndef b():\n    return a().x\n")
    tb = f'  File "{mod}", line 5, in b\n'
    enriched = enrich_traceback(tb, tmp_path)
    assert "return a().x" in enriched
    assert "->" in enriched


def test_enrich_skips_outside_root(tmp_path):
    tb = '  File "/usr/lib/python3/stdlib.py", line 10, in something\n'
    enriched = enrich_traceback(tb, tmp_path)
    assert enriched.strip() == tb.strip()


def test_enrich_passes_through_non_frames(tmp_path):
    tb = "TypeError: expected str, got None\n"
    assert "TypeError" in enrich_traceback(tb, tmp_path)


def test_summarize_trims_long_output():
    detail = "x" * 5000
    out = summarize_failure("pytest", detail, max_chars=100)
    assert "trimmed" in out
    assert len(out) < 400


def test_summarize_labels_check():
    out = summarize_failure("ruff", "F401 unused import")
    assert "[ruff failed]" in out
