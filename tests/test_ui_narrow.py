"""The chat on a narrow phone screen (40 / 54 / 80 columns), run in a pty against a fake Ollama and read back through a terminal emulator:
no markup text, no raw httpx error, no line wider than the screen, every notice once. Needs `pexpect` and `pyte` (skipped without)."""
from __future__ import annotations

import pytest

from tests._ptyscreen import Fake, run_chat

pytestmark = pytest.mark.posix_only


def screen_for(tmp_path, width, *, chat_404, text="hello\r", wait=8.0, expect="next:"):
    fake = Fake(tmp_path, chat_404=chat_404)
    try:
        return run_chat(tmp_path, fake.port, width, [(text, wait)] if text else [], expect=expect)
    finally:
        fake.close()


@pytest.mark.parametrize("width", [40, 54, 80])
def test_404_is_one_friendly_line_and_nothing_overflows(tmp_path, width):
    out = screen_for(tmp_path, width, chat_404=True)
    assert "[red]" not in out and "[/" not in out
    for raw in ("Client error", "developer.mozilla", "For more information", "api/chat", "http://", "HTTPStatusError", "will retry"):
        assert raw not in out, raw
    assert "is not installed" in out and "nightage pull" in out
    assert out.count("is not installed") == 1                       # said once, not twice
    assert all(len(line.rstrip()) <= width for line in out.split("\n"))
    assert out.count("Using qwen3.5:0.8b") <= 1


@pytest.mark.parametrize("width", [40])
def test_start_screen_has_no_markup_and_fits(tmp_path, width):
    out = screen_for(tmp_path, width, chat_404=False, text=None, wait=0)
    assert "[dim]" not in out and "[/" not in out and "Agent is ready" in out
    assert all(len(line.rstrip()) <= width for line in out.split("\n"))


def test_a_normal_answer_shows_up(tmp_path):
    out = screen_for(tmp_path, 54, chat_404=False, wait=8.0, expect="Hello from the fake.")
    assert "Hello from the fake." in out and "is not installed" not in out
