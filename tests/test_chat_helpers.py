"""The shared turn helpers live in agent.chat.helpers; the web engine must not pull in the terminal module."""

import os
import subprocess
import sys


def _fresh(code: str) -> str:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120, check=True).stdout.strip()


def test_turn_engine_does_not_import_tui():
    assert _fresh("import sys, agent.chat.turn; print('agent.tui' in sys.modules)") == "False"


def test_web_runtime_and_deps_do_not_import_tui():
    assert _fresh("import sys, agent.web.agent_runtime, agent.web.deps; print('agent.tui' in sys.modules)") == "False"


def test_helpers_reexported_from_tui_are_the_same_objects():
    from agent import tui
    from agent.chat import helpers

    for name in ("DENIED_RESULT", "_frame_tool_result", "_format_tool_call", "_extract_tool_call", "_num_ctx",
                 "_strip_thinking_artifacts", "_react_to_stuck", "_classify_call", "_make_remember_tool",
                 "_make_take_note_tool", "_normalize_arguments", "RESULT_TRUNCATE_LEN", "DENIAL_FRAMING"):
        assert getattr(tui, name) is getattr(helpers, name), name
