"""Per-project memory: what the agent learned about THIS repo, kept across sessions.

A project is the nearest folder with a project marker above the files being edited (see agent.verify).
Notes are short durable facts - how to run the tests, a convention, a gotcha, a decision - added by the
model with `project_note` or by the harness (the check command that passed). When a project is active
its newest notes go into the context, so a new session in the same repo does not start from zero.
Separate from personal facts and from the dated web notes.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from agent import verify

MAX_NOTES_IN_CONTEXT = 12
MAX_CHARS_IN_CONTEXT = 1400
MAX_NOTE_CHARS = 300
MAX_NOTES_PER_PROJECT = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_notes (
    id INTEGER PRIMARY KEY, project TEXT NOT NULL, note TEXT NOT NULL, kind TEXT DEFAULT 'note',
    created_at REAL, UNIQUE(project, note)
);
"""


def project_key(path: str | Path) -> str | None:
    """Stable key (the project's root folder) for a file or folder, or None if it isn't in a project."""
    p = Path(path).expanduser()
    root = verify.project_root(str(p / "x" if p.is_dir() else p))        # a folder: look from inside it
    return str(root.resolve()) if root else None


class ProjectMemory:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(_SCHEMA)

    def add(self, project: str, note: str, kind: str = "note") -> bool:
        note = " ".join(note.split())[:MAX_NOTE_CHARS]
        if not project or len(note) < 8:
            return False
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO project_notes (project, note, kind, created_at) VALUES (?,?,?,?)",
            (project, note, kind, time.time()),
        )
        self._conn.execute(
            "DELETE FROM project_notes WHERE project = ? AND id NOT IN "
            "(SELECT id FROM project_notes WHERE project = ? ORDER BY created_at DESC LIMIT ?)",
            (project, project, MAX_NOTES_PER_PROJECT),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def notes(self, project: str, limit: int = MAX_NOTES_IN_CONTEXT) -> list[tuple[int, str]]:
        rows = self._conn.execute(
            "SELECT id, note FROM project_notes WHERE project = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (project, limit),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def clear(self, project: str) -> int:
        cur = self._conn.execute("DELETE FROM project_notes WHERE project = ?", (project,))
        self._conn.commit()
        return cur.rowcount

    def system_message(self, project: str) -> str | None:
        used, lines = 0, []
        for _id, note in self.notes(project):
            if used + len(note) > MAX_CHARS_IN_CONTEXT:
                break
            lines.append(f"- {note}")
            used += len(note)
        if not lines:
            return None
        return (f"PROJECT MEMORY for {Path(project).name} ({project}) - facts learned in earlier sessions; "
                "trust them unless the files now say otherwise:\n" + "\n".join(lines))


class ActiveProject:
    """Which project the chat is currently working in (set by `work in this repo`, access commands, edits)."""

    def __init__(self) -> None:
        self.key: str | None = None

    def set_from(self, path: str | Path | None) -> None:
        if path:
            key = project_key(path)
            if key:
                self.key = key


def make_tool(memory: ProjectMemory, active: ActiveProject, console: Any):
    from agent.tools.registry import Tool

    def project_note(note: str, **_: Any) -> str:
        if not active.key:
            return ("Error: no active project yet. Edit a file inside the project, or have the user say "
                    "'work in this repo: <path>', then try again.")
        if memory.add(active.key, note):
            console.print(f"[dim]project note  {' '.join(note.split())[:90]}[/dim]")
            return f"Saved for the project {Path(active.key).name}; it will be in context in future sessions."
        return "Not saved (too short, or already known)."

    return Tool(
        name="project_note",
        description=(
            "Save a durable fact about the CURRENT project for future sessions: how to run or test it, a "
            "convention, a gotcha you hit, a decision and why. One short sentence each. Not for task progress "
            "or temporary state - and not for facts about the user (use remember_this for those)."
        ),
        parameters={"type": "object", "properties": {"note": {"type": "string", "description": "The fact, one sentence."}},
                    "required": ["note"]},
        handler=project_note,
        risk="safe",
    )
