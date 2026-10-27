"""Shared fakes for the chat-UI tests (no model, no network): scripted streaming client, recording tools,
a wired NightageApp + ChatController around a temp Store, and a pipe-input runner for the real dock."""

from __future__ import annotations

import io
import threading
from pathlib import Path

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from agent.memory.store import Store
from agent.tools.registry import Tool, ToolRegistry
from agent.ui.chat_adapter import ChatState

CONFIG = {
    "models": {"chat": "m:1", "background": "b:1", "embed": "e:1"},
    "persona": {"name": "Agent", "style": "You are Agent."},
    "memory": {"recall_k": 6, "context_char_budget": 24000},
    "tools": {"max_iterations": 8},
    "ollama": {"unload_on_exit": True},
}


def nap(s):
    threading.Event().wait(s)


def wait_for(cond, timeout=5.0):
    for _ in range(int(timeout / 0.01)):
        if cond():
            return True
        nap(0.01)
    return False


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def say(text):
    return {"role": "assistant", "content": text, "_prompt_eval_count": 100, "_eval_count": 20, "_eval_ns": 2_000_000_000}


class FakeClient:
    """Streams scripted assistant messages; a `gate` event holds the first model call until released."""

    def __init__(self, script=(), gate: threading.Event | None = None, models=("m:1", "m:2")):
        self.script, self.gate, self.models = list(script), gate, list(models)
        self.calls, self.unloaded, self.seen = 0, 0, []

    def chat_with_tools_stream(self, messages, model, tools, num_ctx=None, on_content=None, on_thinking=None, **kw):
        self.calls += 1
        self.seen.append(list(messages))
        if self.gate is not None and self.calls == 1:
            self.gate.wait(5)
        message = dict(self.script.pop(0) if self.script else say("ok."))
        message.setdefault("_prompt_eval_count", 100)
        message.setdefault("_eval_count", 20)
        text = message.get("content", "")
        for i in range(0, len(text), 4):
            if on_content:
                on_content(text[i:i + 4])
        return message

    def chat_with_tools(self, messages, model, tools, **kw):
        raise AssertionError("buffered call not expected")

    def chat(self, messages, model, stream=True, **kw):
        return iter(["NONE"])

    def embed(self, texts, model=None):
        raise RuntimeError("no embeddings in tests")

    def list_models(self):
        return list(self.models)

    def supports_tools(self, model):
        return True

    def unload_all(self):
        self.unloaded += 1
        return []


def make_registry(ran: list):
    reg = ToolRegistry()
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    reg.register(Tool("echo", "Echo.", schema, lambda x: f"echoed: {x}"))
    reg.register(Tool("sendit", "Caution.", schema, lambda x: ran.append(("sendit", x)) or f"sent {x}", risk="caution"))
    reg.register(Tool("nuke", "Danger.", schema, lambda x: ran.append(("nuke", x)) or f"nuked {x}", risk="dangerous"))
    return reg


def make_state(tmp_path, client, registry=None, config=None, session_id=None):
    from agent import tui

    cfg = config if config is not None else {**CONFIG, "models": dict(CONFIG["models"])}
    store = Store(tmp_path / "agent.db")
    quiet = Console(file=io.StringIO(), force_terminal=False, width=100)
    state = tui.setup_chat(client, cfg, quiet, store, session_id, tool_registry=registry or make_registry([]))
    return state, cfg, store


def make_app(state, cfg, tmp_path, width=100, **kw):
    from agent.ui.chat_app import build_chat_app

    buf = io.StringIO()
    con = Console(file=buf, width=width, force_terminal=False, color_system=None, highlight=False)
    app, ctl = build_chat_app(state, cfg, plain=True, console=con, banner=False, history_path=tmp_path / "hist",
                              exports_dir=tmp_path / "exports", **kw)
    return app, ctl, buf


def roles(store, sid):
    return [(m["role"], m["content"]) for m in store.get_messages(sid)]


class Piped:
    """Run a callable (usually run_chat_ui or app.run) with pipe keystrokes in a thread."""

    def __init__(self, fn):
        self.fn, self.rc, self.err = fn, [], []

    def __enter__(self):
        self._cm = create_pipe_input()
        self.inp = self._cm.__enter__()
        self.t = threading.Thread(target=self._go, daemon=True)
        self.t.start()
        return self

    def _go(self):
        try:
            with create_app_session(input=self.inp, output=DummyOutput()):
                self.rc.append(self.fn(self.inp))
        except BaseException as exc:  # noqa: BLE001
            self.err.append(exc)

    def type(self, text, pause=0.15):
        nap(pause)
        self.inp.send_text(text)

    def __exit__(self, *a):
        self.t.join(8)
        alive = self.t.is_alive()
        self._cm.__exit__(*a)
        assert not alive, "UI did not exit"
        assert not self.err, self.err
