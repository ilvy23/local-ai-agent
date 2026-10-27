"""The `present` tool: spec validation, provenance, events, fallbacks and keeping blocks out of the model's context."""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from agent.memory.store import Store
from agent.research import taint
from agent.tools import present
from agent.tools.present import SpecError, make_present_tool, validate

TABLE = {"columns": [{"key": "name", "label": "Language"}, {"key": "ms", "label": "Speed", "unit": "ms"}],
         "rows": [{"name": "Rust", "ms": 12}, {"name": "Python", "ms": 410}]}
CHART = {"type": "line", "series": [{"name": "GPU A", "points": [[2023, 499], [2024, 459], [2025, 429]]}]}


class _Events(Console):
    def __init__(self):
        super().__init__(file=io.StringIO(), force_terminal=False, width=90, color_system=None)
        self.events = []

    def emit_event(self, event_type, **fields):
        self.events.append({"type": event_type, **fields})


@pytest.fixture(autouse=True)
def _clean_taint():
    taint.clear()
    yield
    taint.clear()


@pytest.mark.parametrize("kind,spec", [
    ("table", TABLE), ("chart", CHART),
    ("diagram", {"source": "flowchart LR\n A --> B"}),
    ("compare", {"options": ["A", "B"], "criteria": [{"name": "Speed", "weight": 3}, {"name": "Quality", "weight": 5}],
                 "scores": [[8, 6], [7, 9]]}),
    ("timeline", {"events": [{"date": "2026-09-14", "title": "Release"}]}),
    ("stats", {"tiles": [{"label": "VRAM", "value": "21.4", "unit": "GB"}]}),
    ("callout", {"level": "warning", "text": "careful"}), ("quote", {"text": "verbatim"}),
    ("steps", {"steps": [{"title": "Install", "command": "uv tool install -e ."}]}),
    ("code", {"language": "python", "code": "print(1)"}), ("math", {"latex": "E=mc^2"}),
    ("diff", {"files": [{"path": "a.py", "hunks": []}]}),
    ("map", {"points": [{"lat": 48.1, "lon": 11.5, "label": "Munich"}]}),
    ("gallery", {"images": [{"url": "/api/files/abc"}]}), ("svg", {"svg": "<svg viewBox='0 0 1 1'><circle r='1'/></svg>"}),
])
def test_every_kind_has_a_valid_example(kind, spec):
    assert validate(kind, json.loads(json.dumps(spec))) is not None


def test_compare_computes_totals_and_a_winner():
    spec = validate("compare", {"options": ["A", "B"], "criteria": [{"name": "Speed", "weight": 3}, {"name": "Quality", "weight": 5}],
                                "scores": [[8, 6], [7, 9]]})
    assert spec["totals"] == [59, 63] and spec["winner"] == 1


@pytest.mark.parametrize("kind,spec,fragment", [
    ("table", {"columns": [{"key": "a"}], "rows": [{"zzz": 1}]}, "column keys"),
    ("table", {"columns": [], "rows": []}, "columns"),
    ("chart", {"type": "sparkles", "series": []}, "type"),
    ("chart", {"type": "bar", "series": [{"name": "x", "points": [["a", "many"]]}]}, "numeric y"),
    ("diagram", {"source": "hello world"}, "Mermaid"),
    ("compare", {"options": ["A", "B"], "criteria": [{"name": "x"}], "scores": [[1]]}, "scores"),
    ("callout", {"level": "loud", "text": "x"}, "level"),
    ("map", {"points": [{"lat": 200, "lon": 0}]}, "lat"),
    ("gallery", {"images": [{"url": "http://insecure.example/x.png"}]}, "https"),
    ("svg", {"svg": "<svg><script>alert(1)</script></svg>"}, "scripts"),
    ("svg", {"svg": "<svg onload='x()'></svg>"}, "scripts"),
    ("svg", {"svg": "<svg><image href='https://x.example/a.png'/></svg>"}, "scripts"),
    ("nonsense", {}, "unknown kind"),
])
def test_bad_specs_are_rejected_with_a_fixable_message(kind, spec, fragment):
    with pytest.raises(SpecError, match=fragment):
        validate(kind, spec)


def test_spec_can_arrive_as_a_json_string_and_html_is_opt_in():
    assert validate("callout", '{"level": "tip", "text": "hi"}')["level"] == "tip"
    with pytest.raises(SpecError, match="switched off"):
        validate("html", {"html": "<p>x</p>"})
    assert validate("html", {"html": "<p>x</p>", "height": 5000}, allow_html=True)["height"] == 900


def test_limits_are_enforced():
    with pytest.raises(SpecError):
        validate("table", {"columns": [{"key": "a"}], "rows": [{"a": i} for i in range(501)]})
    with pytest.raises(SpecError):
        validate("diagram", {"source": "flowchart LR\n" + "A-->B\n" * 5000})


def test_gui_gets_a_block_event_and_the_model_is_told_not_to_repeat_it():
    console = _Events()
    out = make_present_tool(console).handler(kind="table", title="Speed", spec=TABLE)
    (event,) = console.events
    assert event["type"] == "block" and event["kind"] == "table" and event["id"].startswith("b_")
    assert event["spec"]["rows"][0]["name"] == "Rust" and event["provenance"] == "model"
    assert "Do not repeat" in out and "your own knowledge" in out


def test_bad_spec_returns_an_error_to_the_model_and_emits_nothing():
    console = _Events()
    out = make_present_tool(console).handler(kind="chart", spec={"type": "bar", "series": []})
    assert out.startswith("Error") and "present again" in out and not console.events


def test_numbers_found_in_the_sources_keep_their_provenance_and_invented_ones_are_downgraded():
    console = _Events()
    tool = make_present_tool(console)
    taint.add("Prices fell from 499 to 459 and then 429 euros over three years.")
    tool.handler(kind="chart", spec=CHART, source_ids=[1])
    assert console.events[-1]["provenance"] == "sources" and console.events[-1]["source_ids"] == [1]
    invented = {"type": "bar", "series": [{"name": "x", "points": [["a", 499], ["b", 777]]}]}
    out = tool.handler(kind="chart", spec=invented, source_ids=[1])
    assert console.events[-1]["provenance"] == "model" and "not in any source" in out


def test_computed_is_only_believed_for_figures_the_calc_tool_produced():
    from agent.tools.calc import run as calc_run

    console = _Events()
    tool = make_present_tool(console)
    out = tool.handler(kind="table", spec=TABLE, provenance_claim="computed")           # nothing was calculated
    assert console.events[-1]["provenance"] == "model" and "not produced by the calc tool" in out
    calc_run("t = [12, 410]; sum(t)")
    out = tool.handler(kind="table", spec=TABLE, provenance_claim="computed")           # 12 and 410 came out of calc
    assert console.events[-1]["provenance"] == "computed" and "calc tool" not in out


def test_terminal_fallbacks_draw_text(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.tools.diagram.DIAGRAM_DIR", tmp_path)
    console = Console(file=io.StringIO(), force_terminal=False, width=90, color_system=None)
    tool = make_present_tool(console)
    tool.handler(kind="table", title="Speed", spec=TABLE)
    tool.handler(kind="chart", title="Prices", spec=CHART)
    tool.handler(kind="diagram", title="Flow", spec={"source": "flowchart LR\n A --> B"})
    shown = console.file.getvalue()
    assert "Language" in shown and "Rust" in shown and "█" in shown and "Saved diagram" in shown
    assert list(tmp_path.glob("Flow-*.html"))


def test_blocks_are_stored_for_redraw_but_never_fed_back_to_the_model(tmp_path):
    from agent.memory.recall import RENDER_ONLY_ROLES, build_context

    assert {"file_card", "block"} <= RENDER_ONLY_ROLES
    store = Store(tmp_path / "a.db")
    sid = store.create_session()
    store.add_message(sid, "user", "compare them")
    store.add_message(sid, "block", json.dumps({"type": "block", "kind": "table"}))

    class _NoVectors:
        available = False

    msgs = build_context(store, _NoVectors(), None, {"memory": {"context_char_budget": 24000, "recall_k": 6},
                                                    "persona": {"style": "p"}, "models": {"embed": "e"}}, sid, "and now?")
    assert not any("block" in str(m.get("content")) and '"kind"' in str(m.get("content")) for m in msgs)


def test_block_events_are_persisted_by_the_web_emitter(tmp_path):
    import asyncio

    pytest.importorskip("fastapi")
    from agent.web.routes.chat import _persisting_emit

    store = Store(tmp_path / "a.db")
    sid = store.create_session()
    got = []

    async def emit(e):
        got.append(e)

    asyncio.run(_persisting_emit(store, sid, emit)({"type": "block", "kind": "table", "id": "b_1"}))
    assert got and [m["role"] for m in store.get_messages(sid)] == ["block"]
