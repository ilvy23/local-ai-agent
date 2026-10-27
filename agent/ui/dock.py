"""The dock: input field, activity row and footer, built on prompt_toolkit's PromptSession (no full-screen app).

    [live tail of the answer being streamed]      <- only while text is arriving
    [spinner] reading foo.py | 12s | 1.2k tok      <- activity row (what is happening now)
    > _                                            <- the prompt; stays usable while a turn runs
    model | ctx 42% | q2 | ask | ~/proj            <- footer (bottom_toolbar)

Everything printed by the Renderer lands ABOVE this block through `patch_stdout(raw=True)`. The dock never uses
rich.Live; progress is drawn by prompt_toolkit's own refresh (`refresh_interval`)."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from agent import osport
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout

from agent.ui import style
from agent.ui.commands import REGISTRY, Registry

QUIT = object()
SPIN_UNICODE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPIN_ASCII = "-\\|/"
_DOCK_ACTIVE = 0
_ACTIVE_LOCK = threading.Lock()

KEYS_HELP = (
    ("Enter", "send (or queue while the agent is busy)"),
    ("Alt+Enter / Ctrl-J", "new line (bracketed paste works too)"),
    ("Up (empty, busy)", "pull the last queued prompt back"),
    ("Esc / Ctrl-C", "stop the current turn (twice: quit)"),
    ("Ctrl-O", "toggle verbose details"),
    ("Ctrl-L", "clear the screen"),
    ("Tab", "complete /commands and @files"),
    ("?", "this list (on an empty line)"),
    ("y/n/a/e + Enter", "answer an approval: allow, deny, always, explain (typed, never a single key)"),
    ("Ctrl-R", "search history"),
)


def dock_active() -> bool:
    return _DOCK_ACTIVE > 0


def state_dir() -> Path:
    base = str(osport.state_home())
    return Path(base) / "nightage"


# ----------------------------------------------------------------------------- footer & activity row


@dataclass
class FooterState:
    model: str = ""
    backend: str = ""
    gpu: str = ""
    ctx_used: int | None = None
    ctx_limit: int | None = None
    tok_s: float | None = None
    access: str = ""
    folder: str = ""
    queue: int = 0
    queue_paused: bool = False
    agents: int = 0
    mood: str = ""
    state: str = "idle"

    @property
    def ctx_pct(self) -> int | None:
        if self.ctx_used is None or not self.ctx_limit:
            return None
        return round(100 * self.ctx_used / self.ctx_limit)


def _short_folder(path: str, room: int) -> str:
    if not path:
        return ""
    home = str(Path.home())
    p = "~" + path[len(home):] if path == home or path.startswith(home + os.sep) else path
    if len(p) <= room:
        return p
    base = os.path.basename(p.rstrip(os.sep)) or p
    return style.trunc(base, room) if len(base) > room else base if room < len(p) else p


def footer_segments(s: FooterState, width: int, plain: bool = False) -> list[str]:
    """Segments in priority order, already shortened for the layout class of `width`."""
    lay = style.layout(width)
    g = style.glyphs(plain)
    segs: list[str] = []
    if s.model:
        segs.append(style.trunc(s.model, 30 if lay == "wide" else 22 if lay == "normal" else 16, g))
    pct = s.ctx_pct
    if pct is not None:
        if lay == "wide":
            segs.append(f"ctx {style.fmt_tokens(s.ctx_used or 0)}/{style.fmt_tokens(s.ctx_limit or 0)} {pct}%")
        else:
            segs.append(f"ctx {pct}%")
    if s.queue or s.queue_paused:
        segs.append(f"q{s.queue}" + ("!" if s.queue_paused else "") if lay == "narrow" else
                    f"queue {s.queue}" + (" (paused)" if s.queue_paused else ""))
    if s.agents:
        segs.append(f"{s.agents} agent{'s' if s.agents != 1 else ''}" if lay != "narrow" else f"{s.agents}ag")
    if s.access:
        segs.append(s.access)
    if s.tok_s:
        segs.append(f"{s.tok_s:.0f} tok/s")
    if s.backend or s.gpu:
        segs.append(" ".join(x for x in (s.backend, s.gpu) if x))
    if s.folder:
        segs.append(_short_folder(s.folder, 36 if lay == "wide" else 24))
    if s.mood:
        segs.append(s.mood)
    return segs


def fit_footer(s: FooterState, width: int, plain: bool = False) -> str:
    """One line, never wider than `width`; drops low-priority segments first."""
    sep = " | " if plain else " · "
    width = max(10, width)
    out: list[str] = []
    for seg in footer_segments(s, width, plain):
        cand = sep.join([*out, seg])
        if len(cand) <= width - 1:
            out.append(seg)
        elif not out:
            out.append(style.trunc(seg, width - 1, style.glyphs(plain)))
    return sep.join(out)


SLOW_HINT_S = 10.0     # after this long without a first token the row explains the wait


def activity_row(snap: dict[str, Any], width: int, plain: bool = False, tick: float | None = None,
                 verbose: bool = False) -> str:
    """The one-line 'what is happening now' row. `snap` is TurnWorker.snapshot()."""
    g = style.glyphs(plain)
    sep = " | " if plain else " · "
    t = time.time() if tick is None else tick
    spin = (SPIN_ASCII if plain else SPIN_UNICODE)
    if snap.get("busy"):
        state = snap.get("state")
        frame = spin[int(t * 10) % len(spin)]
        if state == "paused":
            lead, now = "||" if plain else "⏸", "paused"
        elif state == "waiting_approval":
            lead, now = "?", snap.get("now") or "waiting for approval"
        else:
            lead, now = frame, snap.get("now") or "working"
        if snap.get("waiting_s", 0) >= SLOW_HINT_S and state not in ("paused", "waiting_approval"):
            est = int(snap.get("prompt_tokens") or 0)
            now = (f"reading the prompt (~{style.fmt_tokens(est)} tokens)" if est else "reading the prompt") \
                + "... on CPU this can take a few minutes"
        bits = [now, style.fmt_duration(snap.get("elapsed", 0))]
        if snap.get("waiting_s", 0) >= SLOW_HINT_S and state not in ("paused", "waiting_approval"):
            bits.reverse()               # the clock first: a long hint may be cut at narrow widths, the time must not be
        if snap.get("tokens"):
            bits.append(f"{style.fmt_tokens(snap['tokens'])} tok")
        if snap.get("tok_s"):
            rate = float(snap["tok_s"])
            bits.append(f"{rate:.0f} tok/s" if rate >= 10 else f"{rate:.1f} tok/s")
        if snap.get("queue"):
            bits.append(f"{snap['queue']} queued")
        text = f"{lead} " + sep.join(bits)
    elif snap.get("queue_paused") and snap.get("queue"):
        text = f"queue paused ({snap['queue']} waiting){sep}/resume to continue"
    else:
        text = "Enter send" + sep + "/ commands" + sep + "@ files" + sep + "? keys" if width >= 50 else "? keys | / commands"
        text = text.replace(" · ", " | ") if plain else text
    return style.trunc(text, max(10, width - 1), g)


# ----------------------------------------------------------------------------- completion


class NightageCompleter(Completer):
    """/command completion (fuzzy, with the summary as meta) and @file path completion."""

    def __init__(self, registry: Registry = REGISTRY, cwd: Callable[[], str] | None = None) -> None:
        self.registry = registry
        self.cwd = cwd or os.getcwd

    def get_completions(self, document, complete_event) -> Iterable[Completion]:  # noqa: ANN001
        before = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)
        if word.startswith("@"):
            yield from self._files(word)
            return
        if before.startswith("/") and " " not in before and "\n" not in before:
            for c in self.registry.completions(before)[:12]:
                yield Completion(c.name, start_position=-len(before), display=c.name,
                                 display_meta=style.trunc(c.summary, 48))

    def _files(self, word: str) -> Iterable[Completion]:
        frag = word[1:]
        base = Path(self.cwd())
        directory, _, prefix = frag.rpartition("/")
        root = (base / directory) if directory else base
        if frag.startswith("/"):
            root = Path(directory or "/")
        elif frag.startswith("~"):
            root = Path(os.path.expanduser(directory or "~"))
        try:
            entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        shown = 0
        for p in entries:
            n = p.name
            if n.startswith(".") and not prefix.startswith("."):
                continue
            if n in ("__pycache__", "node_modules") and not prefix:
                continue
            if prefix and not n.lower().startswith(prefix.lower()):
                continue
            is_dir = p.is_dir()
            text = "@" + (directory + "/" if directory else "") + n + ("/" if is_dir else "")
            yield Completion(text, start_position=-len(word), display=text[1:], display_meta="dir" if is_dir else "")
            shown += 1
            if shown >= 50:
                break


# ----------------------------------------------------------------------------- the dock


@dataclass
class DockHooks:
    """What the dock asks of the application. All optional; the dock is usable (and testable) without an app."""
    on_line: Callable[[str], Any] = lambda text: None
    busy: Callable[[], bool] = lambda: False
    on_stop: Callable[[], Any] = lambda: None
    on_toggle_verbose: Callable[[], Any] = lambda: None
    on_clear: Callable[[], Any] = lambda: None
    on_keys_help: Callable[[], Any] = lambda: None
    pull_back: Callable[[], str | None] = lambda: None
    approval_pending: Callable[[], bool] = lambda: False
    approval_dangerous: Callable[[], bool] = lambda: False
    on_approval_key: Callable[[str], Any] = lambda key: None
    footer: Callable[[], FooterState] = lambda: FooterState()
    activity: Callable[[], dict[str, Any]] = lambda: {}
    live_tail: Callable[[], str] = lambda: ""
    on_notice: Callable[[str], Any] = lambda text: None


class Dock:
    def __init__(self, hooks: DockHooks | None = None, *, registry: Registry = REGISTRY, plain: bool | None = None,
                 history_path: str | os.PathLike | None = None, input: Any = None, output: Any = None,  # noqa: A002
                 cwd: Callable[[], str] | None = None, width: Callable[[], int] | None = None,
                 live_guard: str = "suppress", quit_window: float = 2.0, refresh: float = 0.25) -> None:
        self.hooks = hooks or DockHooks()
        self.plain = style.plain_mode() if plain is None else plain
        self.registry = registry
        self.live_guard = live_guard
        self.quit_window = quit_window
        self._last_ctrl_c = 0.0
        self._width = width
        self._exit_requested = False
        self.history: History = FileHistory(str(history_path)) if history_path else InMemoryHistory()
        if history_path:
            try:
                Path(history_path).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.history = InMemoryHistory()
        self.completer = NightageCompleter(registry, cwd)
        self.session = PromptSession(
            message=self._message, bottom_toolbar=self._toolbar, multiline=True, key_bindings=self._bindings(),
            history=self.history, completer=self.completer, complete_while_typing=True, refresh_interval=refresh,
            enable_history_search=False, mouse_support=False, input=input, output=output, erase_when_done=False,
            prompt_continuation=lambda width, ln, wrap: "  ",
            style=None, reserve_space_for_menu=3,
        )
        # Esc must react quickly; Alt+Enter still works (both are escape sequences)
        self.session.app.ttimeoutlen = 0.05
        self.session.app.timeoutlen = 0.3

    # ---- sizing
    def columns(self) -> int:
        if self._width is not None:
            return self._width()
        try:
            return self.session.app.output.get_size().columns
        except Exception:  # noqa: BLE001
            return 80

    # ---- drawn pieces
    def _message(self) -> FormattedText:
        w = self.columns()
        parts: list[tuple[str, str]] = []
        if self.session.app.is_done:      # final frame: leave only the typed line in the scrollback
            return FormattedText([("class:prompt", "> " if self.plain else "› ")])
        try:
            tail = self.hooks.live_tail()
        except Exception:  # noqa: BLE001
            tail = ""
        if tail:
            parts.append(("class:tail", style.trunc("  " + tail[-(w * 2):], max(10, w - 1)).replace("\n", " ")[: max(10, w - 1)] + "\n"))
        try:
            row = activity_row(self.hooks.activity(), w, self.plain)
        except Exception:  # noqa: BLE001
            row = ""
        parts.append(("class:activity", row + "\n"))
        parts.append(("class:prompt", "> " if self.plain else "› "))
        return FormattedText(parts)

    def _toolbar(self) -> FormattedText:
        w = self.columns()
        try:
            text = fit_footer(self.hooks.footer(), w, self.plain)
        except Exception:  # noqa: BLE001
            text = ""
        return FormattedText([("class:bottom-toolbar", text)])

    # ---- keys
    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()
        h = self.hooks

        empty = Condition(lambda: not self.session.default_buffer.text)
        busy = Condition(lambda: bool(h.busy()))
        can_pull = Condition(lambda: bool(h.busy()) and not self.session.default_buffer.text)

        @kb.add("enter")
        def _enter(event):  # noqa: ANN001
            buf = event.current_buffer
            if not buf.text.strip():
                buf.reset()
                return
            buf.validate_and_handle()

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _newline(event):  # noqa: ANN001
            event.current_buffer.insert_text("\n")

        @kb.add("escape", eager=False)
        def _esc(event):  # noqa: ANN001
            if h.busy():
                h.on_stop()

        @kb.add("c-c")
        def _ctrl_c(event):  # noqa: ANN001
            now = time.monotonic()
            buf = event.current_buffer
            if now - self._last_ctrl_c < self.quit_window:
                event.app.exit(result=QUIT)
                return
            self._last_ctrl_c = now
            if h.busy():
                h.on_stop()
                h.on_notice("stopping - press Ctrl-C again to quit")
            elif buf.text:
                buf.reset()
            else:
                h.on_notice("press Ctrl-C again to quit")

        @kb.add("c-d")
        def _ctrl_d(event):  # noqa: ANN001
            if not event.current_buffer.text:
                event.app.exit(result=QUIT)

        @kb.add("c-o")
        def _ctrl_o(event):  # noqa: ANN001
            h.on_toggle_verbose()

        @kb.add("c-l")
        def _ctrl_l(event):  # noqa: ANN001
            event.app.renderer.clear()
            h.on_clear()

        @kb.add("?", filter=empty & ~Condition(lambda: bool(h.approval_pending())))
        def _qmark(event):  # noqa: ANN001
            h.on_keys_help()

        @kb.add("up", filter=can_pull)
        def _up(event):  # noqa: ANN001
            text = h.pull_back()
            if text is None:
                event.current_buffer.auto_up(count=event.arg)
                return
            event.current_buffer.text = text
            event.current_buffer.cursor_position = len(text)

        return kb

    # ---- run
    def request_exit(self) -> None:
        self._exit_requested = True
        app = self.session.app
        loop = getattr(app, "loop", None)
        if not app.is_running or loop is None:
            return

        def _exit() -> None:
            if app.is_running and not app.future.done():
                app.exit(result=QUIT)

        try:
            loop.call_soon_threadsafe(_exit)
        except Exception:  # noqa: BLE001
            pass

    def read_once(self) -> Any:
        """Prompt once; returns the text, or QUIT. Must run inside `active()`."""
        try:
            return self.session.prompt()
        except (EOFError, KeyboardInterrupt):
            return QUIT

    @contextmanager
    def active(self):
        """patch_stdout(raw=True) + the rich.Live guard. Everything printed inside appears above the dock."""
        global _DOCK_ACTIVE
        restore = _install_live_guard(self.live_guard)
        with _ACTIVE_LOCK:
            _DOCK_ACTIVE += 1
        try:
            with patch_stdout(raw=True):
                yield self
        finally:
            with _ACTIVE_LOCK:
                _DOCK_ACTIVE -= 1
            restore()

    def run(self) -> None:
        """Read lines until QUIT / request_exit; each line goes to hooks.on_line (which must not block long)."""
        with self.active():
            while not self._exit_requested:
                result = self.read_once()
                if result is QUIT or self._exit_requested:
                    break
                try:
                    self.hooks.on_line(result)
                except Exception as exc:  # noqa: BLE001
                    self.hooks.on_notice(f"internal error: {exc}")


# ----------------------------------------------------------------------------- rich.Live guard


def _install_live_guard(mode: str) -> Callable[[], None]:
    """While the dock is active a rich.Live/Status would fight prompt_toolkit for the cursor. `suppress` turns
    Live.start() into a no-op (so legacy code using console.status degrades to silence), `raise` makes it a
    loud error (tests), `off` leaves rich alone."""
    if mode == "off":
        return lambda: None
    from rich.live import Live

    orig = Live.start

    def guarded(self, refresh: bool = False):  # noqa: ANN001
        if mode == "raise":
            raise RuntimeError("rich.Live/console.status must not be used while the dock is active")
        return None

    Live.start = guarded  # type: ignore[method-assign]

    def restore() -> None:
        if Live.start is guarded:
            Live.start = orig  # type: ignore[method-assign]

    return restore
