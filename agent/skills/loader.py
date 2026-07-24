from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    tools: tuple[str, ...] = ()
    path: Path | None = None

    def header(self) -> str:
        return f"{self.name}: {self.description}"


def parse_skill(text: str, path: Path | None = None) -> Skill | None:
    front, body = _split_front_matter(text)
    if front is None:
        return None
    try:
        meta = yaml.safe_load(front) or {}
    except yaml.YAMLError:
        return None
    name = str(meta.get("name", "")).strip()
    description = str(meta.get("description", "")).strip()
    if not name:
        return None
    tools = meta.get("tools") or []
    if not isinstance(tools, list):
        tools = []
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        tools=tuple(str(t) for t in tools),
        path=path,
    )


def _split_front_matter(text: str) -> tuple[str | None, str]:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return None, text
    after = stripped[3:]
    end = after.find("\n---")
    if end == -1:
        return None, text
    front = after[:end]
    body = after[end + 4 :]
    return front, body


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def headers(self) -> str:
        return "\n".join(self._skills[name].header() for name in self.names())

    def load_dir(self, root: Path) -> list[str]:
        loaded = []
        for skill_file in sorted(root.glob("*/SKILL.md")):
            try:
                text = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue
            skill = parse_skill(text, path=skill_file)
            if skill is not None:
                self.register(skill)
                loaded.append(skill.name)
        return loaded
