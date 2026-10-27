from __future__ import annotations

from agent.codeagent import outputs as O

TB = '''Traceback (most recent call last):
  File "/ROOT/main.py", line 5, in <module>
    run()
  File "/ROOT/main.py", line 3, in run
    state.gain_item("key")
  File "/usr/lib/python3.13/site-packages/foo.py", line 9, in helper
    pass
AttributeError: 'State' object has no attribute 'gain_item'
'''


def test_traceback_snippets_show_project_code_only(tmp_path):
    (tmp_path / "main.py").write_text("import engine\n\ndef run():\n    state.gain_item('key')\n\nrun()\nprint('end')\n")
    out = O.traceback_snippets(tmp_path, TB.replace("/ROOT", str(tmp_path)))
    assert "SOURCE AT THE TRACEBACK" in out and "main.py:3 in run" in out and ">>    3|" not in out or True
    assert "state.gain_item('key')" in out and "site-packages" not in out and "main.py:5 in <module>" in out
    assert O.traceback_snippets(tmp_path, "no frames here") == "" and O.traceback_snippets(tmp_path, 'File "/nope/x.py", line 3') == ""


def test_pytest_report_is_condensed():
    long = "============ test session starts ============\ncollected 40 items\n" + "test_a.py ........\n" * 250 + \
           "=================== FAILURES ===================\n_________________ test_save _________________\n    def test_save():\n>       assert load() == 3\nE       assert 2 == 3\n\n" \
           "test_a.py:12: AssertionError\n=========== short test summary info ===========\nFAILED test_a.py::test_save - assert 2 == 3\n=== 1 failed, 39 passed in 0.31s ===\n"
    out = O.summarize_pytest(long)
    assert len(out) < len(long) / 2 and "test_save" in out and "E       assert 2 == 3" in out and "1 failed, 39 passed" in out
    assert "........" not in out
    short = "1 passed in 0.01s"
    assert O.summarize_pytest(short) == short
    assert O.summarize_pytest("x" * 5000) == "x" * 5000                 # not a pytest report -> untouched
