"""Apply parsed edits to a workspace, validating as we go.

A search block that doesn't match the file is a malformed edit, not a silent
no-op — it's reported so the caller re-asks rather than "succeeding" wrongly.
"""

from __future__ import annotations

from agent.coding.edit.format import Edit


def apply_edits(workspace, edits: list[Edit]) -> tuple[list[str], list[str]]:
    """Apply edits. Returns (changed_paths, errors). Any error → treat as malformed."""
    changed: list[str] = []
    errors: list[str] = []
    for edit in edits:
        if edit.whole is not None:
            workspace.write(edit.path, edit.whole if edit.whole.endswith("\n") else edit.whole + "\n")
            changed.append(edit.path)
            continue
        try:
            src = workspace.read(edit.path)
        except FileNotFoundError:
            errors.append(f"{edit.path}: file does not exist")
            continue
        new = src
        for search, replace in edit.replacements:
            if search not in new:
                errors.append(f"{edit.path}: SEARCH block not found — it must match the file exactly")
                continue
            new = new.replace(search, replace, 1)
        if new != src:
            workspace.write(edit.path, new)
            changed.append(edit.path)
    return changed, errors
