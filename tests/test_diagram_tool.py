from pathlib import Path
from unittest.mock import MagicMock

from agent.tools.diagram import make_diagram, make_diagram_tool


def test_make_diagram_writes_html_with_mermaid_source(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent.tools.diagram.DIAGRAM_DIR", tmp_path)
    out = make_diagram(title="Login Flow", diagram="flowchart TD\nA[Start] --> B[End]")
    assert "Error" not in out
    files = list(tmp_path.glob("*.html"))
    assert len(files) == 1
    content = files[0].read_text()
    assert "flowchart TD" in content
    assert "A[Start] --> B[End]" in content
    assert "mermaid" in content.lower()


def test_make_diagram_rejects_empty_content(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent.tools.diagram.DIAGRAM_DIR", tmp_path)
    out = make_diagram(title="Empty", diagram="")
    assert "Error" in out
    assert list(tmp_path.glob("*.html")) == []


def test_make_diagram_sanitizes_title_for_filename(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent.tools.diagram.DIAGRAM_DIR", tmp_path)
    make_diagram(title="a/b:weird*name?", diagram="graph TD\nA --> B")
    files = list(tmp_path.glob("*.html"))
    assert len(files) == 1
    assert "/" not in files[0].name.replace(str(tmp_path), "")


def test_make_diagram_never_overwrites_previous_diagrams(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent.tools.diagram.DIAGRAM_DIR", tmp_path)
    make_diagram(title="Same Title", diagram="graph TD\nA --> B")
    make_diagram(title="Same Title", diagram="graph TD\nC --> D")
    assert len(list(tmp_path.glob("*.html"))) == 2


def test_make_diagram_tool_narrates_success_to_console(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent.tools.diagram.DIAGRAM_DIR", tmp_path)
    console = MagicMock()
    tool = make_diagram_tool(console)
    result = tool.handler(title="My Flow", diagram="graph TD\nA --> B")
    assert "Error" not in result
    console.print.assert_called_once()
    assert "My Flow" in console.print.call_args[0][0]


def test_make_diagram_tool_does_not_narrate_on_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("agent.tools.diagram.DIAGRAM_DIR", tmp_path)
    console = MagicMock()
    tool = make_diagram_tool(console)
    result = tool.handler(title="Empty", diagram="")
    assert "Error" in result
    console.print.assert_not_called()
