import io

from rich.console import Console

from agent import liveview, tui


def _console(width=100):
    return Console(file=io.StringIO(), force_terminal=False, width=width, color_system=None)


def test_approval_is_one_line_and_y_n_only():
    c = _console()
    seen = []
    c.input = lambda prompt="": seen.append(prompt) or "y"
    assert tui._approve(c, tui.RiskLevel.CAUTION, "edit a.py +3 -1")
    assert len(seen) == 1 and "\n" not in seen[0] and "[y/n]" in seen[0].replace("\\[", "[")
    for answer, expected in (("yes", True), ("Y", True), ("n", False), ("a", False), ("", False)):
        c.input = lambda prompt="", a=answer: a
        assert tui._approve(c, tui.RiskLevel.CAUTION, "x") is expected
    c.input = lambda prompt="": "y"
    assert not tui._approve(c, tui.RiskLevel.DANGEROUS, "x")          # DANGEROUS still needs the word yes
    c.input = lambda prompt="": "yes"
    assert tui._approve(c, tui.RiskLevel.DANGEROUS, "x")


def test_describe_file_edit_shows_path_and_delta_never_the_code(tmp_path):
    c = _console()
    f = tmp_path / "demo.py"
    d = tui._describe_call(c, "write_file", {"path": str(f), "content": "print('SECRET_CODE')\n" * 5})
    assert "create" in d and "+5" in d and "-0" in d and "SECRET_CODE" not in d
    f.write_text("a\nb\n")
    d = tui._describe_call(c, "write_file", {"path": str(f), "content": "a\nX\nY\n"})
    assert "overwrite" in d and "+2" in d and "-1" in d
    d = tui._describe_call(c, "replace_in_file", {"path": str(f), "search": "a\nb", "replace": "c"})
    assert "edit" in d and "+1" in d and "-2" in d


def test_describe_shell_command_is_one_truncated_line():
    c = _console(width=60)
    d = tui._describe_call(c, "run_command", {"command": "cd /x &&\n" + "echo hello " * 40})
    assert "\n" not in d and len(d) <= 60 and d.endswith("…")
    assert "[" not in tui._describe_call(c, "run_command", {"command": "echo [bold]"}).replace("\\[", "")


def test_coding_status_is_green_with_animated_dots_and_no_rotating_words():
    clock = iter(range(0, 10_000))
    t = liveview.ThinkingText("coding", style="green", clock=lambda: next(clock) * 0.5)
    first = t.render()
    assert first.plain.startswith("coding.") and str(first.spans[0].style) == "green"
    assert {t.render().plain.split(".")[0] for _ in range(400)} == {"coding"}


def test_editing_status_shows_arrow_and_file():
    t = liveview.ThinkingText("editing", style="green", suffix="rps/game.py")
    assert t.render().plain.startswith("editing.") and "→ rps/game.py" in t.render().plain


def test_looks_like_coding_from_the_ask():
    assert tui._looks_like_coding("Refactor ~/Downloads/proj/rock_paper_scissors.py into a package")
    assert tui._looks_like_coding("schreib mir ein skript das Dateien umbenennt")
    assert not tui._looks_like_coding("what is the weather in Berlin tomorrow?")


def test_thinking_cue_fires_once_on_code_words_split_across_chunks():
    fired = []
    cue = tui._ThinkingCue(lambda: fired.append(1))
    for piece in ["The user wants a game. I should ", "use write", "_file to create it", " and then write_file again"]:
        cue.feed(piece)
    assert fired == [1]
    quiet = tui._ThinkingCue(lambda: fired.append(2))
    quiet.feed("The user asks about the weather in Berlin.")
    assert fired == [1]


def test_spinner_can_switch_to_coding_mid_wait_but_fixed_labels_stay():
    s = tui._ThinkingSpinner(_console())
    assert s._text.label != "coding"
    s.set_coding()
    assert s._text.label == "coding" and s._text.style == "green"
    fixed = tui._ThinkingSpinner(_console(), label="compacting")
    fixed.set_coding()
    assert fixed._text.label == "compacting"


def test_stream_turn_passes_on_thinking_only_to_clients_that_accept_it():
    seen = {}

    class Old:
        def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None):
            seen["old"] = True
            return {"content": ""}

    class New:
        def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None):
            seen["new"] = on_thinking
            return {"content": ""}

    class Live:
        started = False
        on_content = staticmethod(lambda p: None)

    f = lambda p: None
    tui._stream_turn(Old(), [], "m", [], 1, Live(), f)
    tui._stream_turn(New(), [], "m", [], 1, Live(), f)
    assert seen == {"old": True, "new": f}
