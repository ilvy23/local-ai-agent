import json
from io import StringIO

import httpx
from rich.console import Console

from agent.llm import OllamaClient
from agent.memory.store import Store
from agent.safety import RiskLevel
from agent.tools.registry import Tool, ToolRegistry, default_registry
from agent.tui import ChatSession, _approve, _frame_tool_result, _strip_thinking_artifacts, run_repl

CONFIG = {
    "models": {"chat": "mixtral:8x7b", "background": "llama3.1:8b", "embed": "nomic-embed-text"},
    "persona": {"name": "Agent", "style": "You are Agent."},
    "memory": {"recall_k": 6, "context_char_budget": 24000},
    "tools": {"max_iterations": 8},
}


class _ScriptedConsole(Console):
    """A Console whose .input() replays scripted lines then raises EOFError."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__(file=StringIO(), force_terminal=False)
        self._lines = list(lines)

    def input(self, prompt: str = "", **kwargs) -> str:
        self.print(prompt, end="")
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


class FakeClient:
    def __init__(self, reply_tokens):
        self.reply_tokens = reply_tokens
        self.received_messages = None  # messages from the most recent chat call
        self.chat_calls: list[list[dict]] = []

    def chat(self, messages, model, stream=True, **kwargs):
        self.received_messages = list(messages)
        self.chat_calls.append(list(messages))
        yield from self.reply_tokens

    def chat_with_tools(self, messages, model, tools, **kwargs):
        self.received_messages = list(messages)
        self.chat_calls.append(list(messages))
        return {"role": "assistant", "content": "".join(self.reply_tokens)}


class ScriptedToolClient:
    """chat_with_tools() returns each scripted message dict in order.

    Once a tool has been called, _run_agent_turn discards the non-streamed
    wrap-up content from chat_with_tools() and re-requests it via a plain
    streamed chat() call instead (so the user watches it arrive live).
    `chat()` here replays the content of the last scripted message that had
    no tool_calls, so the streamed "final answer" a test asserts on matches
    what was actually scripted, not an unrelated placeholder.
    """

    def __init__(self, messages: list[dict]):
        self._messages = list(messages)
        self.chat_with_tools_calls: list[list[dict]] = []
        self._last_content = ""

    def chat_with_tools(self, messages, model, tools, **kwargs):
        self.chat_with_tools_calls.append(list(messages))
        message = self._messages.pop(0)
        if not message.get("tool_calls"):
            self._last_content = message.get("content", "")
        return message

    def chat(self, messages, model, stream=True, **kwargs):
        yield self._last_content


def test_one_turn_appends_user_and_assistant_messages_to_history():
    client = FakeClient(["Hel", "lo!"])
    session = ChatSession(client=client, model="mixtral:8x7b", system_prompt="You are Agent.")

    reply = session.send("hi there")

    assert reply == "Hello!"
    assert session.history == [
        {"role": "system", "content": "You are Agent."},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_history_sent_to_client_includes_prior_turns():
    client = FakeClient(["ok"])
    session = ChatSession(client=client, model="mixtral:8x7b", system_prompt="sys")

    session.send("first")
    client.reply_tokens = ["second-reply"]
    session.send("second")

    assert client.received_messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]


def test_send_streams_tokens_to_callback():
    client = FakeClient(["a", "b", "c"])
    session = ChatSession(client=client, model="mixtral:8x7b", system_prompt="sys")

    seen = []
    session.send("hi", on_token=seen.append)

    assert seen == ["a", "b", "c"]


def test_run_repl_persists_messages_to_store(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = FakeClient(["Hello", " there!"])
    console = _ScriptedConsole(["hi there", "/quit"])

    run_repl(client, CONFIG, console=console, store=store)

    session_id = store.get_last_session_id()
    assert session_id is not None
    stored = [{"role": m["role"], "content": m["content"]} for m in store.get_messages(session_id)]
    assert stored == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "Hello there!"},
    ]
    store.close()


def test_run_repl_sets_session_title_from_first_message(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = FakeClient(["ok"])
    console = _ScriptedConsole(["hello world", "/quit"])

    run_repl(client, CONFIG, console=console, store=store)

    session_id = store.get_last_session_id()
    sessions = store.list_sessions()
    assert sessions[0]["title"] == "hello world"
    store.close()


def test_run_repl_truncates_long_title_to_60_chars(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = FakeClient(["ok"])
    long_message = "x" * 100
    console = _ScriptedConsole([long_message, "/quit"])

    run_repl(client, CONFIG, console=console, store=store)

    sessions = store.list_sessions()
    assert sessions[0]["title"] == "x" * 60


def test_run_repl_resumes_existing_session_with_history(tmp_path):
    store = Store(tmp_path / "agent.db")
    session_id = store.create_session(title="Old chat")
    store.add_message(session_id, "user", "earlier question")
    store.add_message(session_id, "assistant", "earlier answer")

    client = FakeClient(["fresh reply"])
    console = _ScriptedConsole(["new question", "/quit"])

    run_repl(
        client,
        CONFIG,
        console=console,
        store=store,
        session_id=session_id,
        tool_registry=ToolRegistry(),  # no tools -> system prompt has no tool instructions
    )

    # First chat call is the actual turn (a later call may be fact distillation).
    call = client.chat_calls[0]
    # The system prompt starts with the persona and now also carries a live
    # "System environment" block, so check the prefix rather than an exact match.
    assert call[0]["role"] == "system"
    assert call[0]["content"].startswith("You are Agent.")
    assert call[1:] == [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "new question"},
    ]
    messages = store.get_messages(session_id)
    stored_tail = [{"role": m["role"], "content": m["content"]} for m in messages[-2:]]
    assert stored_tail == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "fresh reply"},
    ]
    store.close()


def _registry_with_echo():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echoes text back.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=lambda text: f"echoed: {text}",
        )
    )
    return registry


def test_scripted_tool_call_executes_and_final_answer_follows(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = ScriptedToolClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "echo", "arguments": {"text": "hi"}}}
                ],
            },
            {"role": "assistant", "content": "The tool said: echoed: hi"},
        ]
    )
    console = _ScriptedConsole(["use the echo tool", "/quit"])

    run_repl(client, CONFIG, console=console, store=store, tool_registry=_registry_with_echo())

    session_id = store.get_last_session_id()
    messages = store.get_messages(session_id)
    roles = [m["role"] for m in messages]
    assert "tool" in roles
    assert {"role": messages[-1]["role"], "content": messages[-1]["content"]} == {
        "role": "assistant", "content": "The tool said: echoed: hi",
    }
    store.close()


def test_model_rejecting_tools_falls_back_to_json_parsed_tool_call_end_to_end(tmp_path):
    """mixtral:8x7b-style 400 "does not support tools" -> transparent retry
    without tools=, and the plain-text JSON tool call still executes via the
    fallback parser, producing a normal final answer."""
    store = Store(tmp_path / "agent.db")
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/api/show":  # the startup capability probe
            return httpx.Response(
                200, json={"capabilities": ["completion", "tools"]}, request=request
            )
        requests_seen.append(body)
        if "tools" in body:
            return httpx.Response(
                400,
                json={"error": "mixtral:8x7b does not support tools"},
                request=request,
            )
        # Second call in the same turn: model replies with the fallback
        # JSON tool-call format instead of structured tool_calls.
        if not any(m["role"] == "tool" for m in body["messages"]):
            content = json.dumps({"tool": "echo", "arguments": {"text": "hi"}})
        else:
            content = "The tool said: echoed: hi"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": content}, "done": True},
            request=request,
        )

    client = OllamaClient(transport=httpx.MockTransport(handler))
    console = _ScriptedConsole(["use the echo tool", "/quit"])

    class _NoVectors:
        available = False

    run_repl(
        client,
        CONFIG,
        console=console,
        store=store,
        vectors=_NoVectors(),
        tool_registry=_registry_with_echo(),
    )

    session_id = store.get_last_session_id()
    messages = store.get_messages(session_id)
    roles = [m["role"] for m in messages]
    assert "tool" in roles
    assert {"role": messages[-1]["role"], "content": messages[-1]["content"]} == {
        "role": "assistant", "content": "The tool said: echoed: hi",
    }
    # First request attempted tools=, the rest did not.
    assert "tools" in requests_seen[0]
    assert all("tools" not in r for r in requests_seen[1:])
    store.close()


def test_tool_handler_exception_is_fed_back_and_loop_continues(tmp_path):
    store = Store(tmp_path / "agent.db")

    def boom(**kwargs):
        raise ValueError("kaboom")

    registry = ToolRegistry()
    registry.register(
        Tool(name="boom", description="Always fails.", parameters={"type": "object"}, handler=boom)
    )
    client = ScriptedToolClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "boom", "arguments": {}}}],
            },
            {"role": "assistant", "content": "Sorry, that failed."},
        ]
    )
    console = _ScriptedConsole(["try the tool", "/quit"])

    run_repl(client, CONFIG, console=console, store=store, tool_registry=registry)

    session_id = store.get_last_session_id()
    messages = store.get_messages(session_id)
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "kaboom" in tool_messages[0]["content"]
    assert {"role": messages[-1]["role"], "content": messages[-1]["content"]} == {
        "role": "assistant", "content": "Sorry, that failed.",
    }
    store.close()


def test_max_iterations_cap_tells_user_and_stops(tmp_path):
    store = Store(tmp_path / "agent.db")
    tool_call_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "echo", "arguments": {"text": "hi"}}}],
    }
    config = {**CONFIG, "tools": {"max_iterations": 2}}
    # Always returns a tool call, never a final answer -> should hit the cap.
    client = ScriptedToolClient([tool_call_message] * 10)
    console = _ScriptedConsole(["loop forever", "/quit"])

    run_repl(client, config, console=console, store=store, tool_registry=_registry_with_echo())

    assert len(client.chat_with_tools_calls) == 2
    console_output = console.file.getvalue()
    assert "step limit" in console_output.lower()
    store.close()


def test_tool_messages_not_embedded(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = ScriptedToolClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "echo", "arguments": {"text": "hi"}}}
                ],
            },
            {"role": "assistant", "content": "final answer"},
        ]
    )
    console = _ScriptedConsole(["use tool", "/quit"])

    class RecordingVectors:
        available = True

        def __init__(self):
            self.added = []

        def add(self, kind, ref_id, text, embedding):
            self.added.append((kind, text))

        def search(self, embedding, k=6, kinds=None):
            return []

        def max_cosine_similarity(self, embedding, kind):
            return 0.0

    class EmbedClient(ScriptedToolClient):
        def embed(self, texts, model):
            return [[0.0] for _ in texts]

    embed_client = EmbedClient(client._messages)
    vectors = RecordingVectors()

    run_repl(
        embed_client,
        CONFIG,
        console=console,
        store=store,
        vectors=vectors,
        tool_registry=_registry_with_echo(),
    )

    embedded_texts = [text for _kind, text in vectors.added]
    assert "use tool" in embedded_texts
    assert "final answer" in embedded_texts
    assert not any("echoed" in t for t in embedded_texts)


class _ConnectErrorClient:
    """chat_with_tools() raises a connection error, as if Ollama is down."""

    def chat_with_tools(self, messages, model, tools, **kwargs):
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        raise httpx.ConnectError("connection refused", request=request)

    def chat(self, messages, model, stream=True, **kwargs):
        yield ""


class _StatusErrorClient:
    """chat_with_tools() raises an unrelated HTTP status error (not the
    tools-unsupported 400), e.g. a 500 from a broken Ollama server."""

    def __init__(self, status_code: int, error_message: str):
        self._status_code = status_code
        self._error_message = error_message

    def chat_with_tools(self, messages, model, tools, **kwargs):
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        response = httpx.Response(
            self._status_code, json={"error": self._error_message}, request=request
        )
        raise httpx.HTTPStatusError(
            f"{self._status_code} error", request=request, response=response
        )

    def chat(self, messages, model, stream=True, **kwargs):
        yield ""


def test_connect_error_shows_connection_hint(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = _ConnectErrorClient()
    console = _ScriptedConsole(["hi", "/quit"])

    run_repl(client, CONFIG, console=console, store=store)

    output = console.file.getvalue()
    assert "Could not reach Ollama" in output
    store.close()


def test_unrelated_http_status_error_is_not_reported_as_connection_error(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = _StatusErrorClient(500, "internal server error")
    console = _ScriptedConsole(["hi", "/quit"])

    run_repl(client, CONFIG, console=console, store=store)

    output = console.file.getvalue()
    assert "Could not reach Ollama" not in output
    assert "500" in output
    assert "internal server error" in output
    store.close()


def test_audit_log_written_for_tool_execution(tmp_path):
    store = Store(tmp_path / "agent.db")
    client = ScriptedToolClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "echo", "arguments": {"text": "hi"}}}
                ],
            },
            {"role": "assistant", "content": "final answer"},
        ]
    )
    console = _ScriptedConsole(["use tool", "/quit"])

    run_repl(client, CONFIG, console=console, store=store, tool_registry=_registry_with_echo())

    rows = store.conn.execute("SELECT kind, detail, approved, result FROM audit_log").fetchall()
    assert len(rows) == 1
    assert rows[0]["kind"] == "tool"
    assert "echo" in rows[0]["detail"]
    assert rows[0]["approved"] == 1
    assert "echoed: hi" in rows[0]["result"]
    store.close()


class TestApprove:
    """The prompt used to display "[y/N]" but only ever accept the exact
    single letter "y" - silently declining the very natural answer "yes".
    CAUTION must now accept y/yes case-insensitively; DANGEROUS stays
    strict on purpose (a deliberate extra step for high-stakes actions like
    sending a real message), so it must NOT loosen the same way."""

    def _approve_with(self, answer: str, level: RiskLevel) -> bool:
        console = _ScriptedConsole([answer])
        return _approve(console, level, "some_tool(arg='value')")

    def test_caution_accepts_y(self):
        assert self._approve_with("y", RiskLevel.CAUTION) is True

    def test_caution_accepts_yes(self):
        assert self._approve_with("yes", RiskLevel.CAUTION) is True

    def test_caution_accepts_yes_any_case(self):
        assert self._approve_with("Yes", RiskLevel.CAUTION) is True
        assert self._approve_with("YES", RiskLevel.CAUTION) is True
        assert self._approve_with("Y", RiskLevel.CAUTION) is True

    def test_caution_rejects_n_and_empty(self):
        assert self._approve_with("n", RiskLevel.CAUTION) is False
        assert self._approve_with("", RiskLevel.CAUTION) is False

    def test_dangerous_accepts_only_exact_lowercase_yes(self):
        assert self._approve_with("yes", RiskLevel.DANGEROUS) is True

    def test_dangerous_still_rejects_y_and_other_case(self):
        # Locks in the intentional asymmetry: DANGEROUS must stay stricter
        # than CAUTION even after the CAUTION fix above.
        assert self._approve_with("y", RiskLevel.DANGEROUS) is False
        assert self._approve_with("Yes", RiskLevel.DANGEROUS) is False
        assert self._approve_with("YES", RiskLevel.DANGEROUS) is False


def test_write_file_result_is_framed_to_stop_content_repetition():
    # A small model asked to "write a script" would otherwise plausibly call
    # write_file, then narrate "here's what I wrote: ```...```" right after -
    # the same content shown twice, looking exactly like a duplicate-render
    # bug from the user's side. The framing must explicitly forbid this.
    framed = _frame_tool_result("write_file", "wrote 42 chars to /tmp/x.py")
    assert "do not paste" in framed.lower() or "not paste" in framed.lower()
    assert "wrote 42 chars to /tmp/x.py" in framed


def test_replace_in_file_result_is_framed_to_stop_content_repetition():
    framed = _frame_tool_result("replace_in_file", "Replaced 1 occurrence in /tmp/x.py")
    assert "do not paste" in framed.lower() or "not paste" in framed.lower()
    assert "Replaced 1 occurrence in /tmp/x.py" in framed


class TestStripThinkingArtifacts:
    """A hybrid-reasoning model (Qwen3.5 on Ollama 0.32.14) was observed once
    putting its reasoning directly in `content` with a trailing orphaned
    </think> marker, instead of Ollama routing it to the separate `thinking`
    field this codebase already correctly ignores. Defensive cleanup, not a
    root-cause fix - the model/template decides where reasoning goes."""

    def test_orphaned_close_tag_keeps_only_the_text_after_it(self):
        leaked = (
            "Berlin is the capital of Germany.\n</think>\n\n"
            "I see you're interested in learning more about Berlin!"
        )
        assert _strip_thinking_artifacts(leaked) == (
            "I see you're interested in learning more about Berlin!"
        )

    def test_properly_paired_think_block_is_removed_entirely(self):
        paired = "<think>\nOkay let me think about this...\n</think>\n\nThe answer is 42."
        assert _strip_thinking_artifacts(paired) == "The answer is 42."

    def test_clean_content_with_no_think_tags_passes_through_unchanged(self):
        clean = "Just a normal reply with no think tags."
        assert _strip_thinking_artifacts(clean) == clean
