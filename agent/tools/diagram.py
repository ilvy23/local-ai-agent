"""Diagram tool: renders a Mermaid diagram as a small, self-contained HTML
file the user opens in a browser. The terminal chat can't draw graphics
itself, so this hands off to the one thing on the user's machine that can -
any modern browser, no extra software needed (Mermaid.js loads from a CDN).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agent.tools.registry import Tool

# A dedicated, fixed folder rather than an arbitrary model-given path - like
# remember_this/take_note, the contained blast radius (always here, always
# timestamped, never overwrites anything) is what makes this safe to run
# without an approval prompt, unlike write_file's arbitrary-path writes.
DIAGRAM_DIR = Path("~/Documents/agent-diagrams").expanduser()

_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; color: #1a1a1a; }}
  h1 {{ font-size: 1.1rem; color: #444; font-weight: 600; }}
  .mermaid {{ background: white; padding: 1.5rem; border-radius: 8px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<pre class="mermaid">
{diagram}
</pre>
<script>mermaid.initialize({{ startOnLoad: true }});</script>
</body>
</html>
"""


def _safe_filename(title: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
    return cleaned or "diagram"


def make_diagram(title: str = "", diagram: str = "", **_kwargs: Any) -> str:
    """Write `diagram` (Mermaid syntax) as a small standalone HTML file the
    user can open in any browser to see it rendered.
    """
    diagram = (diagram or "").strip()
    if not diagram:
        return "Error: diagram content was empty."
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    # Microsecond precision: two diagrams made back-to-back in the same turn
    # would otherwise collide on a same-second timestamp and overwrite.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = DIAGRAM_DIR / f"{_safe_filename(title or 'diagram')}-{stamp}.html"
    html = _HTML_TEMPLATE.format(title=title or "Diagram", diagram=diagram)
    try:
        path.write_text(html, encoding="utf-8")
    except OSError as exc:
        return f"Error writing diagram: {exc}"
    return f"Saved diagram to {path} - open it in a browser to view it rendered."


def make_diagram_tool(console: Any) -> Tool:
    """Build the make_diagram tool bound to `console` so a save is visible
    in the terminal - the plain handler is otherwise silent (risk="safe",
    no approval prompt either), and the model's own wrap-up reply can be
    empty (an observed hybrid-reasoning-model quirk), which would leave no
    sign anything happened at all.
    """

    def handler(title: str = "", diagram: str = "", **_kwargs: Any) -> str:
        result = make_diagram(title=title, diagram=diagram)
        if not result.startswith("Error"):
            console.print(f"[dim]📊 diagram saved: {title or 'diagram'}[/dim]")
        return result

    return Tool(
        name="make_diagram",
        description=(
            "Create a visual diagram to explain a process, architecture, "
            "relationship, or flow - use this whenever a picture would make "
            "something clearer than prose or a bullet list (e.g. how requests "
            "move through a system, the steps in a process, how pieces relate "
            "to each other). Write the diagram in Mermaid syntax (flowchart, "
            "sequence diagram, etc.) - it's saved as a file the user opens in "
            "a browser to see it rendered, since a terminal can't display "
            "graphics directly. Do not also describe the whole diagram in "
            "prose afterward - just say what it shows and where it was saved."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the diagram."},
                "diagram": {
                    "type": "string",
                    "description": (
                        "The diagram in Mermaid syntax, e.g. "
                        "'flowchart TD\\nA[Start] --> B[End]'."
                    ),
                },
            },
            "required": ["title", "diagram"],
        },
        handler=handler,
        risk="safe",
    )
