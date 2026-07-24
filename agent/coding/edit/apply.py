from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.coding.edit.format import Edit, EditKind


@dataclass(frozen=True)
class ApplyResult:
    changed: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def _resolve_within(root: Path, rel: str) -> Path | None:
    root = root.resolve()
    candidate = (root / rel).resolve()
    if candidate == root or root in candidate.parents:
        return candidate
    return None


def apply_edits(edits: tuple[Edit, ...] | list[Edit], root: Path) -> ApplyResult:
    root = Path(root)
    planned: list[tuple[Path, str, str]] = []
    errors: list[str] = []

    for edit in edits:
        abs_path = _resolve_within(root, edit.path)
        if abs_path is None:
            errors.append(f"refusing edit outside sandbox: {edit.path}")
            continue

        if edit.kind is EditKind.WHOLE_FILE:
            planned.append((abs_path, edit.replace, edit.path))
            continue

        if not abs_path.exists():
            errors.append(f"search/replace target does not exist: {edit.path}")
            continue
        try:
            current = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"cannot read {edit.path}: {exc}")
            continue

        assert edit.search is not None
        occurrences = current.count(edit.search)
        if occurrences == 0:
            errors.append(f"SEARCH text not found in {edit.path}")
            continue
        if occurrences > 1:
            errors.append(
                f"SEARCH text appears {occurrences} times in {edit.path}; "
                "make it unique with more surrounding context"
            )
            continue
        planned.append((abs_path, current.replace(edit.search, edit.replace, 1), edit.path))

    if errors:
        return ApplyResult(changed=(), errors=tuple(errors))

    changed: list[str] = []
    for abs_path, new_text, rel in planned:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(new_text, encoding="utf-8")
        if rel not in changed:
            changed.append(rel)

    return ApplyResult(changed=tuple(changed))
