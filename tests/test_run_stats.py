"""scripts/run_stats.py on a small synthetic events log + saved session. No model, no home dir."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("run_stats", Path(__file__).resolve().parent.parent / "scripts" / "run_stats.py")
RS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RS)

T0 = 1_700_000_000.0


def _write(tmp_path):
    ev = []

    def add(dt, kind, **kw):
        ev.append({"ts": T0 + dt, "session": "proj-abcd1234", "kind": kind, **kw})

    add(5, "step", n=1, prompt=1000, gen=40, prefill_s=3.0, decode_s=2.0)   # first call precedes first event
    add(5, "tool", name="read_file", seconds=0.5, ok=True)
    add(10, "step", n=2, prompt=3000, gen=30, prefill_s=1.0, decode_s=1.0, load_s=0.0, think=False)
    add(10, "tool", name="edit_file", seconds=1.0, ok=True)
    add(15, "step", n=3, prompt=5000, gen=30, prefill_s=1.0, decode_s=1.0)
    add(15, "tool", name="run", seconds=2.0, ok=False)
    add(16, "loop", tool="run", shape="same", repeats=3)
    add(17, "look_blocked", tool="read_file", streak=25)
    add(18, "stage_cap", id=1, title="x", seconds=10.0, prompt_peak=5000)
    add(20, "loop_abort", tool="run", repeats=6)
    add(2000, "step", n=4, prompt=2000, gen=10, prefill_s=1.0, decode_s=1.0)  # > 5 min later -> idle
    add(2000, "tool", name="search", seconds=0.1, ok=True)
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in ev) + "\nnot json\n")
    sd = tmp_path / "sessions" / "hash"
    sd.mkdir(parents=True)
    (sd / "abcd1234.meta.json").write_text(json.dumps({"id": "abcd1234", "title": "T", "state": "interrupted", "stopped": "loop", "model": "m"}))
    (sd / "abcd1234.json").write_text(json.dumps({"messages": [], "engine_state": {"declined": [["run", "h"]]}}))
    return p, tmp_path / "sessions"


def test_analyse(tmp_path):
    p, sd = _write(tmp_path)
    rows = RS.analyse("proj-abcd1234", RS.load_events(p), RS.load_metas(sd), 300.0)
    r = rows
    assert r["steps"] == 4 and r["tool_calls"] == 4

    assert r["idle_gaps"] == 1
    assert 1977 < r["idle_s"] < 1981
    assert r["prompt_peak"] == 5000 and r["prompt_mean"] == 2750
    assert (r["reads"], r["edits"], r["runs"]) == (2, 1, 1)
    assert r["read_edit_ratio"] == 2.0
    assert r["tool_failed"] == 1
    assert r["loops_by_tool"] == {"run": 1} and r["aborts_by_tool"] == {"run": 1}
    assert r["look_blocked"] == 1 and r["stage_cap"] == 1
    assert r["declined_saved"] == 1
    assert r["outcome"].startswith("interrupted")
    # wall = 5 (lead-in) + 15 (events 5..20) + 2 (own model time of the step after the idle gap) 
    assert abs(r["wall_s"] - 22.0) < 0.2
    assert r["share_pct"]["prefill"] > 0 and r["prefill_s"] == 6.0


def test_cli_json_since_and_session(tmp_path, capsys):
    p, sd = _write(tmp_path)
    base = ["--events", str(p), "--sessions-dir", str(sd)]
    assert RS.main(base + ["--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["overall"]["sessions"] == 1 and d["sessions"][0]["session"] == "proj-abcd1234"
    assert RS.main(base + ["--session", "nope"]) == 0
    assert "no matching" in capsys.readouterr().out
    assert RS.main(base + ["--since", "1h"]) == 0   # synthetic events are far in the past
    assert "no matching" in capsys.readouterr().out
    assert RS.main(base) == 0
    assert "OVERALL" in capsys.readouterr().out


def test_parse_since():
    assert RS.parse_since(None) is None
    assert RS.parse_since("24h") is not None
