"""agent/resume.py: the registry of every resumable run (coding sessions + nights), `continue`, `night runs`. Never touches the real home."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from rich.console import Console

from agent import resume as R
from agent.codeagent import sessionstore as SS


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _session(root: Path, sid: str, title="build a thing", updated=None, state="finished") -> None:
    SS.save({"id": sid, "root": str(root), "title": title, "created": 1.0, "updated": updated or time.time(), "state": state, "messages": [{"role": "user", "content": "x"}],
             "requests": [title], "step": 3, "model": "m"})


def _night(root: Path, left=3600, statuses=("done", "pending"), over=False) -> None:
    nd = root / ".night"
    nd.mkdir(parents=True, exist_ok=True)
    items = [{"name": f"p{i}", "text": f"prompt text {i}", "status": s, "attempts": 1, "seconds": 5, "note": "", "weight": 2.0 if i == 0 else 1.0, "depends": [], "verify": "", "kind": "", "plan": False}
             for i, s in enumerate(statuses)]
    (nd / "state.json").write_text(json.dumps({"left_s": left, "items": items, "events": [], "flaky": []}))
    if over:
        (nd / "DONE").write_text("x")


def test_registry_merges_sessions_index_and_scanned_nights_newest_first(tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    for d in (a, b, c):
        d.mkdir()
    _session(a, "aaaa1111", updated=time.time() - 500)
    _night(b)                                              # no index line: found by scanning (b is a session folder too)
    _session(b, "bbbb2222", updated=time.time() - 900)
    _night(c)
    R.record_night(c, ["do the work"], {"review": True})   # indexed
    es = R.collect()
    kinds = sorted((e.kind, Path(e.folder).name) for e in es)
    assert kinds == [("code", "a"), ("code", "b"), ("night", "b"), ("night", "c")]
    assert [e.when for e in es] == sorted((e.when for e in es), reverse=True)
    n = next(e for e in es if e.kind == "night" and e.folder == str(c))
    assert n.status == "interrupted" and n.resumable and "1/2 done" in n.desc and n.how.endswith(n.id)


def test_index_is_one_line_per_folder_and_idempotent(tmp_path):
    f = tmp_path / "proj"
    f.mkdir()
    R.record_night(f, ["inline task"], {"review": False})
    first = R._index()[str(f)]
    R.record_night(f, ["something else"], {"review": False}, resumed=True)          # a resume keeps the original sources and start
    R.record_night(f, ["something else"], {"review": False}, resumed=True)
    lines = R.index_path().read_text().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["sources"] == first["sources"] == [{"text": "inline task"}] and d["started"] == first["started"]


def test_status_running_finished_gone_and_state_gone(tmp_path, monkeypatch):
    run, fin, gone, nostate = (tmp_path / n for n in ("run", "fin", "gone", "nostate"))
    for d in (run, fin, nostate):
        d.mkdir()
    _night(run)
    _night(fin, statuses=("done",), over=True)
    for d in (run, fin, gone, nostate):
        R.record_night(d, ["t"], {})
    monkeypatch.setattr("agent.night.lock.owner", lambda f: 4242 if f == run else None)
    st = {Path(e.folder).name: e for e in R.collect()}
    assert st["run"].status == "running" and st["run"].running and not st["run"].resumable
    assert st["fin"].status == "finished" and not st["fin"].resumable
    assert st["gone"].status == "folder gone" and st["gone"].gone
    assert st["nostate"].status == "state gone" and st["nostate"].gone


def test_table_shows_number_kind_folder_status_age_description(tmp_path):
    a = tmp_path / "proj"
    a.mkdir()
    _session(a, "cafe0001", title="fix the parser", updated=time.time() - 7200)
    con = Console(width=200, record=True)
    con.print(R.table(R.collect()))
    out = con.export_text()
    assert "1" in out and "code" in out and "cafe0001" in out and "proj" in out and "finished" in out and "2h" in out and "fix the parser" in out


def test_continue_list_never_asks_and_find_by_number_or_prefix(tmp_path, capsys):
    a = tmp_path / "a"
    a.mkdir()
    _session(a, "abcd1234")
    assert R.continue_cli(list_only=True, ask=lambda q: pytest.fail("asked")) == 0
    es = R.collect()
    assert R.find("1", es)[0] is es[0] and R.find("abc", es)[0] is es[0] and R.find("zzz", es)[0] is None


def test_continue_code_session_dispatches_in_its_folder(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    _session(a, "abcd1234")
    seen = []
    monkeypatch.setattr(R, "_run_code", lambda e: seen.append((e.folder, e.id)) or 0)
    assert R.continue_cli("abcd") == 0 and seen == [(str(a), "abcd1234")]
    assert R.continue_cli(None, ask=lambda q: "1") == 0 and len(seen) == 2             # the interactive pick


def test_continue_night_builds_the_resume_command_with_saved_options(tmp_path, monkeypatch):
    f = tmp_path / "proj"
    f.mkdir()
    pf = tmp_path / "prompt.md"
    pf.write_text("do it")
    _night(f)
    R.record_night(f, [str(pf)], {"review": True, "tournament": 2, "pause_on_game": False, "map_mode": "off", "plan_first": True, "stage_cap": 0, "ctx": 0})
    got = []
    monkeypatch.setattr(R, "_run_night", lambda argv: got.append(argv) or 0)
    assert R.continue_cli(R.night_id(f)) == 0
    argv = got[0]
    assert argv[:2] == ["run", str(pf)] and argv[argv.index("--folder") + 1] == str(f) and "--resume" in argv and "--yes" in argv
    assert "--review" in argv and "--ignore-games" in argv and "--plan" in argv and argv[argv.index("--tournament") + 1] == "2" and argv[argv.index("--map") + 1] == "off"


def test_night_without_index_or_with_moved_prompts_is_rebuilt_from_the_state(tmp_path, monkeypatch):
    f = tmp_path / "proj"
    (f / "x").mkdir(parents=True)
    _night(f)
    _session(f, "dddd0001")                                # makes the folder known, no index line
    got = []
    monkeypatch.setattr(R, "_run_night", lambda argv: got.append(argv) or 0)
    n = next(e for e in R.collect() if e.kind == "night")
    assert R.resume_entry(n, Console()) == 0
    files = [a for a in got[0][1:got[0].index("--folder")]]
    assert len(files) == 2
    from agent.night import queue

    items, _ = queue.load(files)
    assert [(i.name, i.text, i.weight) for i in items] == [("p0", "prompt text 0", 2.0), ("p1", "prompt text 1", 1.0)]


def test_running_night_and_running_session_are_refused_politely(tmp_path, monkeypatch, capsys):
    f = tmp_path / "proj"
    f.mkdir()
    _night(f)
    R.record_night(f, ["t"], {})
    monkeypatch.setattr("agent.night.lock.owner", lambda folder: 99)
    monkeypatch.setattr(R, "_run_night", lambda argv: pytest.fail("must not start"))
    assert R.continue_cli(R.night_id(f)) == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "running right now" in out and "night stop" in out and str(f) in out
    _session(f, "eeee0001", state="in progress")
    monkeypatch.setattr(SS, "folder_busy", lambda root: True)
    monkeypatch.setattr(R, "_run_code", lambda e: pytest.fail("must not start"))
    assert R.continue_cli("eeee") == 1


def test_gone_folder_is_refused_and_prune_removes_dead_entries_only(tmp_path, capsys):
    live, dead = tmp_path / "live", tmp_path / "dead"
    live.mkdir()
    dead.mkdir()
    _session(live, "live0001")
    _night(live)
    _session(dead, "dead0001")
    _night(dead)
    R.record_night(dead, ["t"], {})
    R.record_night(live, ["t"], {})
    import shutil

    shutil.rmtree(dead)
    assert R.continue_cli("dead") == 1 and "no longer exists" in " ".join(capsys.readouterr().out.split())
    assert R.continue_cli(prune_it=True) == 0
    left = R.collect()
    assert sorted(e.id for e in left if e.kind == "code") == ["live0001"] and str(dead) not in R._index() and str(live) in R._index()


def test_night_runs_command_lists_only_nights(tmp_path, capsys):
    from typer.testing import CliRunner

    from agent.night.cli import night_app

    a = tmp_path / "a"
    a.mkdir()
    _session(a, "sess0001", title="a coding session")
    _night(a)
    res = CliRunner().invoke(night_app, ["runs"])
    assert res.exit_code == 0 and "night" in res.output and "sess0001" not in res.output


def test_damaged_index_lines_are_skipped(tmp_path):
    R.index_path().parent.mkdir(parents=True)
    R.index_path().write_text('not json\n{"folder": 5}\n')
    assert R._index() == {} and R.collect() == []


def test_the_list_hides_entries_whose_folder_is_gone_unless_asked(monkeypatch, capsys):
    alive = R.Entry(kind="night", id="n1", folder="/tmp/alive", when=time.time(), status="interrupted", desc="alive one", how="x")
    dead = R.Entry(kind="night", id="n2", folder="/tmp/dead", when=time.time(), status="folder gone", desc="entry xyz", how="x", gone=True, resumable=False)
    monkeypatch.setattr(R, "collect", lambda warn=None: [alive, dead])
    R.continue_cli(list_only=True)
    out = capsys.readouterr().out
    assert "alive one" in out and "entry xyz" not in out and "1 entries whose folder is gone" in out and "hidden" in out
    R.continue_cli(list_only=True, show_all=True)
    assert "entry xyz" in capsys.readouterr().out


def test_inner_coding_sessions_of_a_night_are_hidden_and_not_continued_next_to_a_running_night(monkeypatch, capsys, tmp_path):
    from agent.night import lock as lock_mod

    night = R.Entry(kind="night", id="n1", folder=str(tmp_path), when=time.time(), status="interrupted", desc="the night", how="x")
    inner = R.Entry(kind="code", id="c1", folder=str(tmp_path), when=time.time(), status="interrupted (killed)", desc="inner attempt", how="x", extra={"night_attempt": True})
    plain = R.Entry(kind="code", id="c2", folder="/tmp/elsewhere", when=time.time(), status="interrupted", desc="a normal session", how="x")
    monkeypatch.setattr(R, "collect", lambda warn=None: [night, inner, plain])
    R.continue_cli(list_only=True)
    out = capsys.readouterr().out
    assert "the night" in out and "a normal session" in out and "inner attempt" not in out and "inner sessions of nights" in out
    R.continue_cli(list_only=True, show_all=True)
    assert "inner attempt" in capsys.readouterr().out
    (tmp_path / ".night").mkdir()
    (tmp_path / ".night" / "LOCK").write_text("1")                                                   # a live night in that folder
    assert lock_mod.owner(tmp_path) == 1
    ran = []
    monkeypatch.setattr(R, "_run_code", lambda e: ran.append(e) or 0)
    assert R.resume_entry(inner, Console()) == 1 and ran == []
    (tmp_path / ".night" / "LOCK").unlink()
    assert R.resume_entry(inner, Console()) == 0 and ran == [inner]
