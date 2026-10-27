"""Spill instead of truncate (deepseek-harness `spill` idea): every large tool result is kept whole in .scratch/spill/<tool>-<n>.txt, the model
gets a head+tail preview and ONE precise locator line, so it never re-runs a command to see the middle."""

from __future__ import annotations

import re
import subprocess

from agent.codeagent import spill as S
from agent.codeagent.context import MARKERS, new_markers
from tests.test_codeagent_engine import FakeLLM, _isolated, call, engine, say  # noqa: F401 - _isolated is an autouse fixture

BIG = "\n".join(f"line {i:04d} " + "x" * 40 for i in range(800))       # ~42k chars
LOCATOR = re.compile(r"full output: (\.scratch/spill/(\w+-\d+)\.txt) \((\d+) lines\)[^\n]*read it with read_file path=\S+ start=(\d+) end=(\d+)")


def spill_files(repo):
    d = repo / ".scratch" / "spill"
    return sorted(d.glob("*.txt")) if d.is_dir() else []


def test_a_small_text_is_left_alone(tmp_path):
    sp = S.Spill(tmp_path)
    assert sp.clip("web_fetch", "short\nresult") == "short\nresult" and not spill_files(tmp_path)


def test_a_40k_text_becomes_head_tail_and_one_locator_line(tmp_path):
    sp = S.Spill(tmp_path)
    out = sp.clip("web_fetch", BIG)
    assert len(out) < 7000 and out.startswith("line 0000") and "line 0799" in out and "line 0400" not in out
    m = LOCATOR.search(out)
    assert m and out.count("full output:") == 1 and m.group(3) == "800"
    assert (tmp_path / m.group(1)).read_text() == BIG + "\n"
    # the range named in the line starts right after the shown head and is a real, readable window
    shown_head = int(m.group(4)) - 1
    assert out.splitlines()[shown_head - 1].startswith(f"line {shown_head - 1:04d}")


def test_read_file_of_the_spill_file_works_with_ranges_and_is_not_spilled_again(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    out = eng.execute("run", {"command": "python3 -c \"print('\\n'.join(f'line {i:04d} ' + 'x' * 40 for i in range(800)))\""})
    m = LOCATOR.search(out)
    assert m, out[-600:]
    rel, a, b = m.group(1), int(m.group(4)), int(m.group(5))
    got = eng.execute("read_file", {"path": rel, "start": a, "end": b})
    assert f"--- lines {a}-{b} ---" in got and f"line {a - 1:04d}" in got and "full output:" not in got and "truncated" not in got
    mid = eng.execute("read_file", {"path": rel, "start": 400, "end": 402})
    assert "line 0399" in mid and "line 0401" in mid


def test_run_replay_40k_output_gets_head_tail_and_the_locator(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    res = eng.tools.run(command="python3 -c \"print('\\n'.join(f'line {i:04d} ' + 'x' * 40 for i in range(800)))\"")
    assert res.startswith("exit code 0") and "line 0000" in res and "line 0799" in res and "line 0400" not in res
    assert len(res) < 6500 and LOCATOR.search(res) and "do not re-run" in res
    assert len(spill_files(repo)) == 1 and "line 0400" in spill_files(repo)[0].read_text()
    assert not (repo / ".scratch" / "last_run.txt").exists()


def test_a_repeated_identical_command_does_not_create_new_files(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    cmd = "python3 -c \"print('\\n'.join(f'line {i:04d} ' + 'x' * 40 for i in range(800)))\""
    first = eng.tools.run(command=cmd)
    for _ in range(5):
        again = eng.tools.run(command=cmd)
    assert len(spill_files(repo)) == 1 and LOCATOR.search(first).group(1) == LOCATOR.search(again).group(1)
    eng.tools.run(command=cmd.replace("800", "801"))
    assert len(spill_files(repo)) == 2                                          # different output: a new numbered file


def test_the_directory_is_capped_at_the_newest_30(tmp_path):
    sp = S.Spill(tmp_path)
    paths = [sp.save("run", f"{i}\n" * 3000)[0] for i in range(45)]
    kept = sorted(p.name for p in (tmp_path / ".scratch" / "spill").glob("*.txt"))
    assert len(kept) == S.KEEP == 30
    assert (tmp_path / paths[-1]).is_file() and not (tmp_path / paths[0]).exists()


def test_cleanup_runs_at_stage_start_and_keeps_the_newest(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    d = repo / ".scratch" / "spill"
    d.mkdir(parents=True)
    import os
    for i in range(40):
        f = d / f"run-{i + 1}.txt"
        f.write_text("x\n")
        os.utime(f, (1000 + i, 1000 + i))
    eng.begin_stage("goal", "VERIFIED STATE\nnothing")
    left = sorted(p.name for p in d.glob("*.txt"))
    assert len(left) == 30 and "run-40.txt" in left and "run-1.txt" not in left


def test_numbering_continues_after_a_restart(tmp_path):
    S.Spill(tmp_path).save("run", "a\n" * 4000)
    p2 = S.Spill(tmp_path).save("run", "b\n" * 4000)[0]
    assert p2.endswith("run-2.txt")


def test_the_spill_dir_is_excluded_from_git(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    S.Spill(tmp_path).clip("web_fetch", BIG)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=tmp_path, capture_output=True, text=True).stdout.split()
    assert staged == ["a.py"]


def test_the_stub_passes_the_marker_guard(tmp_path):
    sp = S.Spill(tmp_path)
    out = sp.clip("web_fetch", BIG)
    stub = [ln for ln in out.splitlines() if "full output:" in ln or "not shown" in ln]
    assert len(stub) == 2
    names = [n for n, _rx in MARKERS]
    assert "spill_locator" in names and "spill_gap" in names
    for ln in stub:
        assert new_markers(None, ln), ln                                         # the guard recognises both harness lines
    # nothing else the guard knows is in the preview (those markers have their own emitters)
    assert set(new_markers(None, out)) == {"spill_locator", "spill_gap"}


def test_a_write_that_copies_the_stub_is_refused(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    out = S.Spill(repo).clip("web_fetch", BIG)
    line = LOCATOR.search(out).group(0)
    res = eng.execute("write_file", {"path": "notes.py", "content": f"# {line}\nx = 1\n"})
    assert "harness marker" in res and not (repo / "notes.py").exists()


def test_a_masked_old_command_result_keeps_the_locator(tmp_path):
    from agent.codeagent.context import _stub_for

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    res = eng.tools.run(command="python3 -c \"print('\\n'.join(f'line {i:04d} ' + 'x' * 40 for i in range(800)))\"")
    stub = _stub_for("run", {"command": "x"}, res, repo)
    assert stub.startswith("[old command output trimmed]") and LOCATOR.search(stub)


def test_search_with_too_many_matches_is_spilled_in_full(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    for f in range(20):                                                          # rg keeps 8 matches per file: 20 files give 160 lines
        (repo / f"m{f:02d}.py").write_text("\n".join(f"needle = {i}" for i in range(30)) + "\n")
    res = eng.tools.search(pattern="needle")
    m = LOCATOR.search(res)
    assert m and "more matches" in res and m.group(3) == "160"
    text = (repo / m.group(1)).read_text()
    assert text.count("\n") == 160 and text.startswith(res.splitlines()[0])
    assert f"start={40 + 1} " in res and res.count("\n") < 45


def test_search_results_over_the_threshold_are_cut_at_it_and_spilled(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    for f in range(40):
        (repo / f"m{f:02d}.py").write_text(f"needle = '{'y' * 150}'\n")
    res = eng.tools.search(pattern="needle")
    assert len(res) < 7000 and LOCATOR.search(res) and "more matches" in res
    assert len(spill_files(repo)) == 1 and (repo / LOCATOR.search(res).group(1)).read_text().count("\n") == 40


def test_a_small_search_creates_nothing(tmp_path):
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "m.py").write_text("needle = 1\n")
    assert "full output" not in eng.tools.search(pattern="needle") and not spill_files(repo)


def test_a_big_web_result_goes_through_the_generic_path(tmp_path):
    from agent.tools.registry import Tool

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    eng.registry.register(Tool(name="web_fetch", description="d", parameters={"type": "object", "properties": {"url": {"type": "string"}}},
                               handler=lambda **kw: BIG, risk="safe"))
    eng._prompt_fails = 2                                                        # a web lookup is allowed now
    res = eng.execute("web_fetch", {"url": "https://example.org"})
    assert LOCATOR.search(res) and "web_fetch-1.txt" in res and len(res) < 7000


def test_read_files_and_read_file_keep_their_own_locators_and_are_not_spilled(tmp_path):
    """Their text IS a repo file: line ranges and `continue with start=` already say where to go on; a spill copy would only duplicate source."""
    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    for n in range(4):
        (repo / f"f{n}.py").write_text("\n".join(f"v{i} = {i}  # " + "z" * 60 for i in range(400)) + "\n")
    res = eng.execute("read_files", {"paths": [f"f{n}.py" for n in range(4)]})
    assert res.count("continue with start=") == 4 and "full output:" not in res and not spill_files(repo)


def test_a_failed_write_keeps_the_original_result(tmp_path, monkeypatch):
    sp = S.Spill(tmp_path)
    monkeypatch.setattr(S.Spill, "save", lambda self, tool, text: None)
    assert sp.clip("web_fetch", BIG) == BIG


def test_a_long_single_line_is_wrapped_so_ranges_work(tmp_path):
    sp = S.Spill(tmp_path)
    out = sp.clip("web_fetch", "q" * 50_000)
    m = LOCATOR.search(out)
    assert m and int(m.group(3)) > 100 and len(out) < 7000
    assert all(len(ln) <= S.WRAP for ln in (tmp_path / m.group(1)).read_text().splitlines())


def test_the_prompt_and_the_run_tool_describe_the_spill_dir(tmp_path):
    from agent.codeagent.engine import SYSTEM

    assert ".scratch/spill" in SYSTEM and "last_run" not in SYSTEM
