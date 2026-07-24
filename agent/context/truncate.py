from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

HEAD_LINES = 40
TAIL_LINES = 40


@dataclass(frozen=True)
class Truncated:
    text: str
    truncated: bool = False
    spill_path: Path | None = None
    original_lines: int = 0

    @property
    def was_truncated(self) -> bool:
        return self.truncated


def truncate_output(
    output: str,
    *,
    spill_dir: Path | None = None,
    head: int = HEAD_LINES,
    tail: int = TAIL_LINES,
    ref_id: str | None = None,
) -> Truncated:
    lines = output.splitlines()
    if len(lines) <= head + tail:
        return Truncated(text=output, original_lines=len(lines))

    spill_path = None
    if spill_dir is not None:
        spill_dir.mkdir(parents=True, exist_ok=True)
        ref_id = ref_id or uuid.uuid4().hex[:12]
        spill_path = spill_dir / f"{ref_id}.txt"
        spill_path.write_text(output, encoding="utf-8")

    hidden = len(lines) - head - tail
    marker = f"[... {hidden} lines truncated"
    if spill_path is not None:
        marker += f"; full output at {spill_path}"
    marker += " ...]"

    kept = [*lines[:head], marker, *lines[-tail:]]
    return Truncated(
        text="\n".join(kept),
        truncated=True,
        spill_path=spill_path,
        original_lines=len(lines),
    )
