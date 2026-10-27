"""No hidden output ceilings: replies, tool calls and file reads grow with the chosen context."""

import io
import json

import httpx
from rich.console import Console

from agent import gpumode, llm, tui
from agent.tools import files


def _payload_of_chat(num_ctx):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
        return httpx.Response(200, json={})

    client = llm.OllamaClient(transport=httpx.MockTransport(handler))
    client.chat_with_tools([{"role": "user", "content": "x"}], "m:1", [], num_ctx=num_ctx)
    return seen["options"]


def test_output_cap_grows_with_the_context_window():
    assert llm.output_token_cap(16384) == 4096
    assert llm.output_token_cap(65536) == 16384
    assert llm.output_token_cap(204800) == 32768          # capped
    assert llm.output_token_cap(4096) == 2048             # never below the old default
    assert llm.output_token_cap(None) == 2048


def test_env_sets_or_removes_the_output_cap(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_OUTPUT", "9000")
    assert llm.output_token_cap(16384) == 9000
    monkeypatch.setenv("AGENT_MAX_OUTPUT", "0")
    assert llm.output_token_cap(16384) is None


def test_tool_turn_sends_the_scaled_cap_to_ollama():
    options = _payload_of_chat(65536)
    assert options["num_predict"] == 16384 and options["num_ctx"] == 65536


def test_a_request_above_the_ceiling_is_now_capped_for_every_model_not_just_qwen():
    # agent.llm.cap_ctx used to only clamp qwen-named models; it's generic now (agent.ctxbudget),
    # so even a plainly-named model ("m:1") gets capped at the default ceiling (95000).
    options = _payload_of_chat(204800)
    assert options["num_ctx"] == 95000


def test_unlimited_output_sends_no_num_predict(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_OUTPUT", "-1")
    assert "num_predict" not in _payload_of_chat(16384)


def test_read_file_default_cap_is_the_old_one_without_a_big_context():
    assert files.default_read_cap() == files.READ_MAX_CHARS


def test_read_file_default_cap_follows_a_big_context():
    gpumode._state["ctx"] = 204800
    assert files.default_read_cap() == int(204800 * 1.2)


def test_a_long_file_is_read_completely_when_the_context_allows_it(tmp_path):
    doc = tmp_path / "long.md"
    doc.write_text("zeile\n" * 20000)                     # 120 000 chars, far above the old 20 000
    assert "[truncated]" in files.read_file(str(doc))     # default context: still the old limit
    gpumode._state["ctx"] = 204800
    full = files.read_file(str(doc))
    assert "[truncated]" not in full and full.count("zeile") == 20000


def test_an_explicit_max_chars_is_still_honoured(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("x" * 5000)
    assert "first 100 of 5000" in files.read_file(str(doc), max_chars=100)


# ---- reads are free but visible; writes ask ------------------------------------------------


class _Store:
    def add_audit_log(self, **_k):
        pass


def _run(name, arguments, answer="n"):
    console = Console(file=io.StringIO(), width=100, force_terminal=False)
    asked = []
    console.input = lambda prompt="", **_k: asked.append(prompt) or answer
    result = tui._execute_tool(tui.default_registry(), _Store(), console, {}, name, arguments)
    return result, console.file.getvalue() + "".join(asked)


def test_reading_a_file_runs_without_asking_and_is_shown(tmp_path):
    doc = tmp_path / "n.txt"
    doc.write_text("hallo")
    result, out = _run("read_file", {"path": str(doc)})
    assert "hallo" in result and "read ›" in out and str(doc) in out


def test_listing_a_directory_is_shown_too(tmp_path):
    _, out = _run("list_dir", {"path": str(tmp_path)})
    assert "read ›" in out and "list_dir" in out


def test_writing_still_needs_approval_and_denial_writes_nothing(tmp_path):
    target = tmp_path / "out.md"
    result, out = _run("write_file", {"path": str(target), "content": "x"}, answer="n")
    assert result == tui.DENIED_RESULT and not target.exists() and "Run this?" in out


def test_approved_write_goes_through(tmp_path):
    target = tmp_path / "out.md"
    _run("write_file", {"path": str(target), "content": "inhalt"}, answer="y")
    assert target.read_text() == "inhalt"
