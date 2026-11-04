"""A long spec given to the coding agent as `@prompt.md`: the task is the file's text, the stage planner reads all of it, every
stage is told where the full spec is, and the loops seen in a real run (a symbol #hash refused for a few of its lines, web searches
for the agent's own tool errors, "fixing" EOFError of an interactive program run without input) are gone."""

from __future__ import annotations

import re

from agent import mentions
from agent.codeagent import engine as E
from agent.codeagent import stages
from tests.test_codeagent_editproto import tools as make_tools
from tests.test_codeagent_engine import FakeLLM
from tests.test_codeagent_hardening import _isolated  # noqa: F401
from tests.test_codeagent_hardening import engine as make_engine


def test_a_lone_file_is_the_code_task_and_names_the_file(tmp_path):
    (tmp_path / "prompt.md").write_text("MASTER PROMPT\nBuild a game.\n")
    exp = mentions.expand("@prompt.md", cwd=tmp_path, limit=50_000, as_task=True)
    assert exp.text.startswith("MASTER PROMPT\nBuild a game.") and "the file prompt.md in the project folder" in exp.text


def test_the_planner_reads_the_whole_spec_and_every_stage_is_told_where_it_is(tmp_path):
    spec = "MASTER PROMPT\n" + "".join(f"## {i}. Section\n" + "detail " * 400 + "\n" for i in range(1, 21))
    assert len(spec) > 50_000
    (tmp_path / "prompt.md").write_text(spec)
    task = mentions.expand("@prompt.md", cwd=tmp_path, limit=200_000, as_task=True).text
    head = stages._overall(task)
    assert head.startswith("Overall task: MASTER PROMPT") and "the file prompt.md" in head and "\n\n" not in head
    seen = {}

    class LLM:
        def chat(self, messages, **kw):
            seen["user"] = messages[-1]["content"]
            return iter(['{"stages": []}'])

    stages._model_split(LLM(), {}, task)
    assert seen["user"] == task[:stages.MODEL_TASK_CHARS] and len(seen["user"]) > 50_000
    assert "(full task text" not in stages._overall("fix the login bug")


SRC = ("class Game:\n    def show_menu(self):\n        print('menu')\n        choice = input('> ')\n"
       "        if choice == '1':\n            self.state = 'play'\n")


def test_a_symbol_hash_also_allows_editing_a_few_of_its_lines_while_the_symbol_is_unchanged(tmp_path):
    t = make_tools(tmp_path, src=SRC, name="main.py")
    h = re.search(r"#([0-9a-f]{6,})", t.show_symbol(name="Game.show_menu", path="main.py")).group(1)
    res = t.replace_lines(path="main.py", start=4, end=4, new_text="        choice = '1'", expect=f"#{h}")
    assert "not the lines you saw" not in res and "choice = '1'" in (tmp_path / "main.py").read_text()
    again = t.replace_lines(path="main.py", start=3, end=3, new_text="        print('MENU')", expect=f"#{h}")
    assert "not the lines you saw" in again and "print('menu')" in (tmp_path / "main.py").read_text()


def test_own_tool_errors_are_never_looked_up_on_the_web():
    for msg in ["error: edit_file needs path plus search and replace (exact old text -> new text)",
                "Error: invalid arguments for tool 'run': missing required argument command",
                "Error: no such tool 'run_file'", "lines 35-39 are not the lines you saw (#598b52; now #58a29a)",
                "edit_file NOT APPLIED: this edit would break the syntax of main.py"]:
        assert E._OWN_TOOL_ERROR.search(msg), msg
    assert not E._OWN_TOOL_ERROR.search("ModuleNotFoundError: No module named 'curses'")


def test_an_interactive_program_run_without_input_gets_a_hint_not_a_fix(tmp_path):
    eng = make_engine(tmp_path, FakeLLM([]))[0]
    assert "waits for keyboard input" in eng.tools.run(command="python3 -c \"input('> ')\"")
    assert "waits for keyboard input" not in eng.tools.run(command="python3 -c \"print(input())\"", stdin="hi\n")
