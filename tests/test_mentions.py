"""agent.mentions: `@prompt.md` adds the whole file to the message (chat and `nightage code`), a long spec is never restated in
full every few tool calls, and nothing that is not a text file is touched."""

from __future__ import annotations

from agent import mentions
from agent.chat import gates


def spec_text() -> str:
    """A ~25 kB multi-section markdown spec, like a master development prompt."""
    parts = ["# MASTER DEVELOPMENT PROMPT\n\nBuild a complete terminal visual novel.\n"]
    for i in range(1, 21):
        parts.append(f"\n---\n\n## {i}. Section {i}\n\n" + "".join(
            f"- Requirement {i}.{k}: keep `state` in sync, handle 'quotes', \"double\", $vars and @mentions.\n" for k in range(14)))
    return "".join(parts)


def test_only_a_file_mention_makes_the_file_the_request_and_keeps_every_character(tmp_path):
    spec = spec_text()
    assert len(spec) > 20_000
    (tmp_path / "prompt.md").write_text(spec, encoding="utf-8")
    exp = mentions.expand("@prompt.md", cwd=tmp_path, limit=200_000)
    assert exp.files == [("prompt.md", len(spec))] and not exp.notes
    assert exp.text.startswith(mentions.ONLY_FILES)
    assert f'<file name="prompt.md" chars="{len(spec)}">\n{spec}\n</file>' in exp.text


def test_text_around_the_mention_is_kept_and_mentions_inside_the_file_are_not_expanded(tmp_path):
    (tmp_path / "prompt.md").write_text("follow @other.md\n", encoding="utf-8")
    (tmp_path / "other.md").write_text("SECRET", encoding="utf-8")
    exp = mentions.expand("please build this: @prompt.md.", cwd=tmp_path, limit=50_000)
    assert exp.text.startswith("please build this: @prompt.md.\n\n") and "follow @other.md" in exp.text
    assert "SECRET" not in exp.text and [n for n, _ in exp.files] == ["prompt.md"]


def test_emails_unknown_names_and_binary_files_stay_as_typed(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\0\1\2" * 100)
    text = "mail me@example.com about @nothing-here and @blob.bin"
    exp = mentions.expand(text, cwd=tmp_path, limit=50_000)
    assert exp.text == text and exp.files == [] and any("not a text file" in n for n in exp.notes)


def test_quoted_paths_with_spaces_and_crlf_files(tmp_path):
    (tmp_path / "my spec.md").write_bytes(b"line one\r\nline two\r\n")
    exp = mentions.expand('do @"my spec.md"', cwd=tmp_path, limit=50_000)
    assert "line one\nline two\n" in exp.text and exp.files == [("my spec.md", 18)]


def test_a_file_bigger_than_the_context_is_cut_and_the_cut_is_marked(tmp_path):
    (tmp_path / "big.md").write_text("x" * 30_000, encoding="utf-8")
    exp = mentions.expand("@big.md", cwd=tmp_path, limit=25_000)
    assert "x" * 25_000 in exp.text and "x" * 25_001 not in exp.text and "cut here" in exp.text
    assert any("first 25,000 of 30,000" in n for n in exp.notes)


def test_the_limit_follows_the_chosen_context(monkeypatch):
    from agent import config as cfgmod
    from agent.memory import recall

    monkeypatch.setattr(recall, "history_char_budget", lambda config: 209_715)
    assert mentions.limit_for(cfgmod.DEFAULT_CONFIG) == int(209_715 * mentions.SHARE)
    monkeypatch.setattr(recall, "history_char_budget", lambda config: 1000)
    assert mentions.limit_for(cfgmod.DEFAULT_CONFIG) == mentions.MIN_LIMIT


def test_goal_reminder_never_restates_a_long_spec_in_full():
    spec = spec_text()
    g = {"total_calls": 9, "every_n": 10, "original_task": spec}
    msg = gates.GoalReminder(g).tick().message["content"]
    assert len(msg) < gates.GOAL_REMINDER_CHARS + 300 and "full task is in the conversation above" in msg
    short = {"total_calls": 9, "every_n": 10, "original_task": "fix the login bug"}
    assert "Original task: fix the login bug\n" in gates.GoalReminder(short).tick().message["content"]
