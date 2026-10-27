"""read_file without a range is capped (coding.read_max_chars, default 9000 chars / 250 lines) and says where to go on. Measured: text-rewrite
tasks were 70 % read_file calls, whole 37 KB files filled every fresh stage context to its cap before anything was edited."""

from __future__ import annotations

from agent.codeagent import profile as PR, tools as T
from agent.codeagent.tools import CodeTools

from tests.test_codeagent_context import cm


def tools_for(tmp_path, **kw):
    return CodeTools(tmp_path, PR.detect(tmp_path), cm(tmp_path), **kw)


def numbered(n, width=30):
    return "".join(f"line {i:04d} " + "x" * (width - 10) + "\n" for i in range(1, n + 1))


def test_a_small_file_is_still_read_whole(tmp_path):
    (tmp_path / "a.txt").write_text(numbered(40))
    out = tools_for(tmp_path).read_file("a.txt")
    assert "line 0040" in out and "continue with" not in out


def test_a_long_file_returns_the_first_lines_and_one_continue_line(tmp_path):
    (tmp_path / "big.txt").write_text(numbered(773))
    t = tools_for(tmp_path)
    out = t.read_file("big.txt")
    assert len(out) <= T.READ_MAX_CHARS + 300
    last = int(out.rsplit("lines 1-", 1)[1].split(" of ")[0])
    assert 200 <= last <= 250 and f"line {last:04d}" in out and f"line {last + 1:04d}" not in out
    assert out.endswith(f"lines 1-{last} of 773; continue with start={last + 1}")
    assert out.count("continue with") == 1 and t.read_lines == last
    nxt = t.read_file("big.txt", start=last + 1, end=last + 40)
    assert f"line {last + 1:04d}" in nxt and "continue with" not in nxt


def test_few_very_long_lines_hit_the_char_cap_not_the_line_cap(tmp_path):
    (tmp_path / "prose.md").write_text("".join(f"paragraph {i} " + "word " * 400 + "\n" for i in range(30)))
    out = tools_for(tmp_path).read_file("prose.md")
    assert len(out) <= T.READ_MAX_CHARS + 300 and "of 30; continue with start=" in out


def test_the_cap_is_configurable_and_a_range_has_its_own_larger_cap(tmp_path):
    (tmp_path / "big.txt").write_text(numbered(773, width=60))
    small = tools_for(tmp_path, read_max_chars=3000).read_file("big.txt")
    assert len(small) < 3600 and "continue with start=" in small
    rng = tools_for(tmp_path).read_file("big.txt", start=1, end=200)               # an explicit range: up to 14000 chars
    assert len(rng) > T.READ_MAX_CHARS and "line 0200" in rng and "continue with" not in rng
    huge = tools_for(tmp_path).read_file("big.txt", start=1, end=700)
    assert len(huge) <= T.READ_CAP + 300 and "[truncated - use a smaller range]" in huge


def test_read_files_applies_the_cap_to_each_file(tmp_path):
    (tmp_path / "a.txt").write_text(numbered(600))
    (tmp_path / "b.txt").write_text(numbered(600))
    out = tools_for(tmp_path).read_files(["a.txt", "b.txt"])
    assert out.count("continue with start=") == 2 and len(out) < 2 * (T.READ_MAX_CHARS + 400)


def test_edits_still_work_on_lines_seen_in_a_partial_read(tmp_path):
    (tmp_path / "big.py").write_text("".join(f"X{i} = {i}\n" for i in range(1, 801)))
    t = tools_for(tmp_path)
    out = t.read_file("big.py")
    assert "continue with start=" in out
    res = t.replace_lines("big.py", 5, 5, "X5 = 'five'")
    assert not res.startswith(("error", "NOT CHANGED")) and "X5 = 'five'" in (tmp_path / "big.py").read_text()


def test_the_model_facing_description_mentions_ranges(tmp_path):
    reg = tools_for(tmp_path).registry()
    desc = reg.get("read_file").description
    assert "start/end" in desc and "250 lines" in desc


def test_engine_passes_coding_read_max_chars(tmp_path):
    from tests.test_codeagent_engine import FakeLLM
    from tests.test_codeagent_stages import cfg, new_engine

    eng, _repo, _o = new_engine(tmp_path, FakeLLM([]), cfg(read_max_chars=4000))
    assert eng.tools.read_max_chars == 4000
    eng2, _r, _o2 = new_engine(tmp_path, FakeLLM([]), cfg())
    assert eng2.tools.read_max_chars == T.READ_MAX_CHARS
