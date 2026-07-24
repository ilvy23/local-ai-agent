from pathlib import Path

from agent.skills.loader import SkillRegistry, parse_skill

SKILL_TEXT = """---
name: code-review
description: Review a diff for bugs.
tools: [read_file, run_tests]
---
# Body
Do the review.
"""


def test_parse_skill():
    skill = parse_skill(SKILL_TEXT)
    assert skill.name == "code-review"
    assert skill.description == "Review a diff for bugs."
    assert skill.tools == ("read_file", "run_tests")
    assert "Do the review." in skill.body


def test_parse_no_front_matter():
    assert parse_skill("# just a heading") is None


def test_parse_missing_name():
    assert parse_skill("---\ndescription: x\n---\nbody") is None


def test_header_is_terse():
    skill = parse_skill(SKILL_TEXT)
    assert skill.header() == "code-review: Review a diff for bugs."


def test_registry_two_tier():
    reg = SkillRegistry()
    reg.register(parse_skill(SKILL_TEXT))
    assert "code-review" in reg.names()
    assert "code-review: Review a diff for bugs." in reg.headers()
    assert "Do the review." in reg.get("code-review").body


def test_load_dir(tmp_path):
    d = tmp_path / "code-review"
    d.mkdir()
    (d / "SKILL.md").write_text(SKILL_TEXT)
    reg = SkillRegistry()
    loaded = reg.load_dir(tmp_path)
    assert loaded == ["code-review"]


def test_repo_skill_parses():
    root = Path(__file__).resolve().parent.parent / "skills"
    reg = SkillRegistry()
    loaded = reg.load_dir(root)
    assert "code-review" in loaded
