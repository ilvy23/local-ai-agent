"""A filename that isn't valid UTF-8 must not kill the chat.

Filenames are bytes; nothing guarantees they're UTF-8. Python surfaces
undecodable bytes as lone surrogates, and those raise the moment anything
encodes the string as UTF-8 — like sending a tool result to Ollama as JSON.
A single music folder named `Alizée` written in latin-1 (b'Aliz\\xe9e') took
down a whole session.
"""

from __future__ import annotations

import pytest

from agent.tools.files import list_dir, safe_text, search_files

# What os.listdir() hands back for the latin-1 bytes b'Aliz\xe9e'.
SURROGATE_NAME = "Aliz\udce9e"


def test_the_exact_name_that_crashed_a_session():
    with pytest.raises(UnicodeEncodeError):  # the raw name is a landmine
        SURROGATE_NAME.encode("utf-8")
    cleaned = safe_text(SURROGATE_NAME)
    cleaned.encode("utf-8")  # must not raise
    assert cleaned == "Alizée"  # and the legacy byte is read back correctly


def test_safe_text_leaves_normal_text_alone():
    for s in ("plain", "Alizée", "日本語", "emoji 🎵", ""):
        assert safe_text(s) == s


def _make_latin1_dir(tmp_path):
    """A real directory whose entry name is undecodable as UTF-8."""
    try:
        (tmp_path / b"Aliz\xe9e".decode("utf-8", "surrogateescape")).mkdir()
    except (OSError, UnicodeError):  # pragma: no cover - filesystem won't allow it
        pytest.skip("filesystem rejects non-UTF-8 names")


def test_listing_a_directory_with_a_legacy_name_is_sendable(tmp_path):
    _make_latin1_dir(tmp_path)
    (tmp_path / "normal.mp3").write_text("x")

    out = list_dir(str(tmp_path))
    out.encode("utf-8")  # the crash was here
    assert "- Alizée/ (directory)" in out
    assert "- normal.mp3 (file, 1 bytes)" in out


def test_searching_past_a_legacy_name_is_sendable(tmp_path):
    _make_latin1_dir(tmp_path)
    (tmp_path / "notes.txt").write_text("needle here")

    out = search_files("needle", str(tmp_path))
    out.encode("utf-8")
    assert "notes.txt" in out


def test_tool_results_are_sanitised_before_reaching_the_model():
    """Backstop: the file tools clean their own output, but no tool result at
    all should be able to break the session."""
    from agent.tui import _execute_tool
    from agent.tools.registry import Tool, ToolRegistry

    class _Store:
        def add_audit_log(self, **_kw):
            pass

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="rogue",
            description="returns an unencodable string",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda **_kw: f"found {SURROGATE_NAME}",
            risk="safe",
        )
    )
    from rich.console import Console

    result = _execute_tool(registry, _Store(), Console(), {}, "rogue", {})
    result.encode("utf-8")  # must not raise
    assert "Alizée" in result
