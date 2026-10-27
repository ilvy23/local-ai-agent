"""server_bench with a fake Ollama (httpx.MockTransport): no real server is ever contacted."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location("server_bench", Path(__file__).resolve().parents[1] / "scripts" / "server_bench.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)

CODE_TEXT = "def merge_intervals(x):\n    return sorted(x)\n" * 5


class Fake:
    """Fake server: rate depends on draft; optional text change for a given draft."""

    def __init__(self, change_for_draft: int | None = None):
        self.requests: list[httpx.Request] = []
        self.change_for_draft = change_for_draft

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "m:1"}]})
        body = json.loads(request.content)
        o = body["options"]
        text = CODE_TEXT if "merge_intervals" in body["messages"][0]["content"] else '{"tool_calls": []}'
        if self.change_for_draft == o["draft_num_predict"] and "merge_intervals" in body["messages"][0]["content"]:
            text = "totally different output " * 20
        speed = 40.0 + o["draft_num_predict"] * 2  # tok/s
        final = {"done": True, "prompt_eval_count": 1000, "prompt_eval_duration": int(1e9 * 1000 / (500 + o["num_batch"] / 4)),
                 "eval_count": 100, "eval_duration": int(1e9 * 100 / speed), "draft_accepted": 90}
        lines = [json.dumps({"message": {"content": text}, "done": False}), json.dumps({"message": {"content": ""}, **final})]
        return httpx.Response(200, content="\n".join(lines).encode())


def run(argv, fake=None):
    fake = fake or Fake()
    lines: list[str] = []
    rc = sb.main(argv, transport=httpx.MockTransport(fake), out=lines.append)
    return rc, "\n".join(lines), fake


def test_parse_grid_and_baseline_first():
    assert sb.parse_int_list("4, 6,8,6", "draft") == [4, 6, 8]
    with pytest.raises(SystemExit):
        sb.parse_int_list("a,b", "draft")
    with pytest.raises(SystemExit):
        sb.parse_int_list("0", "batch")
    grid = sb.build_grid([6, 4], [1024, 512])
    assert grid[0] == (4, 512) and len(grid) == 4 and len(set(grid)) == 4
    assert sb.build_grid([8], [2048])[0] == (4, 512)


def test_math():
    assert sb.tps(100, 2e9) == 50.0
    assert sb.tps(0, 1) is None and sb.tps(5, 0) is None
    assert sb.median_of([3.0, 1.0, 2.0]) == 2.0
    assert sb.median_of([1.0, 4.0]) == 2.5
    assert sb.speedup(60.0, 40.0) == 1.5 and sb.speedup(None, 40.0) is None
    assert sb.similarity("abc", "abc") == 1.0 and sb.similarity("abc", "xyz") < 0.5


def test_synthetic_prompt_deterministic_and_prefix_changes():
    a, b = sb.synthetic_source(2000, variant=0), sb.synthetic_source(2000, variant=0)
    assert a == b
    c = sb.synthetic_source(2000, variant=3)
    assert a.splitlines()[0] != c.splitlines()[0]
    assert a.split("\n", 1)[1] == c.split("\n", 1)[1]
    assert len(a) > 2000 * 3


def test_dry_run_sends_nothing():
    rc, out, fake = run(["--dry-run", "--model", "m:1", "--url", "http://x:1"])
    assert rc == 0 and fake.requests == []
    assert "dry run" in out and "4/512" in out and "draft_num_predict" in out


def test_yes_required():
    rc, out, fake = run(["--model", "m:1", "--url", "http://x:1"])
    assert rc == 2 and fake.requests == [] and "--yes" in out


def test_full_run_table_json_and_recommendation(tmp_path):
    path = tmp_path / "o.json"
    rc, out, fake = run(["--yes", "--model", "m:1", "--url", "http://x:1", "--draft", "4,8", "--batch", "512,2048",
                         "--repeats", "2", "--json", str(path), "--temperature-list", "0"])
    assert rc == 0
    assert fake.requests[0].url.path == "/api/ps" and "loaded on the server: m:1" in out
    data = json.loads(path.read_text())
    assert len(data["results"]) == 4 and data["results"][0]["draft"] == 4 and data["results"][0]["batch"] == 512
    base = data["results"][0]
    assert base["speedup"]["code"] == pytest.approx(1.0)
    r8 = next(r for r in data["results"] if (r["draft"], r["batch"]) == (8, 2048))
    assert r8["speedup"]["code"] == pytest.approx(56 / 48)
    assert r8["workloads"]["prefill"]["median"] > base["workloads"]["prefill"]["median"]
    assert base["workloads"]["code"]["draft_stats"] == {"draft_accepted": 90}
    assert data["recommendation"]["draft"] == 8 and data["recommendation"]["batch"] == 2048
    assert "Recommendation: draft_num_predict=8 num_batch=2048" in out
    assert data["determinism"][0]["similarity"] == 1.0
    # every request carried the knobs and a fixed seed / T0
    chats = [json.loads(r.content) for r in fake.requests if r.url.path == "/api/chat"]
    assert all(c["options"]["seed"] == sb.SEED and c["options"]["temperature"] == 0.0 for c in chats)
    assert {(c["options"]["draft_num_predict"], c["options"]["num_batch"]) for c in chats} == {(4, 512), (4, 2048), (8, 512), (8, 2048)}


def test_similarity_guard_flags_changed_output():
    rc, out, _ = run(["--yes", "--model", "m:1", "--url", "http://x:1", "--draft", "4,8", "--batch", "512", "--repeats", "1"],
                     Fake(change_for_draft=8))
    assert rc == 0
    assert "DIFFERS from baseline" in out
    # fastest config is flagged, so it must not be recommended
    assert "Recommendation: draft_num_predict=4 num_batch=512" in out


def test_server_error_returns_1():
    def boom(request):
        return httpx.Response(500, json={"error": "x"})
    lines: list[str] = []
    rc = sb.main(["--yes", "--model", "m", "--url", "http://x:1", "--draft", "4", "--batch", "512"],
                 transport=httpx.MockTransport(boom), out=lines.append)
    assert rc == 1
