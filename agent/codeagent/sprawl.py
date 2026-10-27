"""Early stop of a generation that sprawls: thousands of tokens of prose or thinking and no tool call.

Seen live (task: fix a crashing snake game): step 2 was ONE generation of ~16000 tokens, it ran into the output limit and cost ~5 minutes at
54 tok/s. A real write_file of a big file is long too, so length alone cannot tell them apart: what matters is whether a tool call (JSON
object, `<tool_call>` / `<function=` block) or a code fence is OPEN or was closed within the last `quiet` tokens. A call that is being
written counts as activity for as long as it is open, so a 7000-token write_file is never stopped here (only by the hard cap, NUM_PREDICT).

Pieces stand for tokens (the stream delivers about one piece per token). Pure and cheap: the engine feeds every piece of one generation."""

from __future__ import annotations

import re

SPRAWL_TOKENS = 6000       # a generation longer than this ...
SPRAWL_QUIET = 2000        # ... with no tool call / code fence open or closed in the last this many tokens is stopped

_CALL_START = re.compile(r'\{\s*"(?:tool|name|function)"')
_XML_START = re.compile(r"<tool_call>|<function=")
_XML_END = re.compile(r"</tool_call>|</function>")
_WINDOW = 48               # characters kept to find a marker that is split between two pieces


class SprawlGuard:
    def __init__(self, limit: int = SPRAWL_TOKENS, quiet: int = SPRAWL_QUIET) -> None:
        self.limit, self.quiet = limit, quiet
        self.n = 0                  # pieces of this generation (content and thinking)
        self.active_at = 0          # the piece number of the last structure activity
        self._tail = ""
        self._depth = 0             # > 0: inside an open JSON call (braces outside strings)
        self._in_str = self._esc = False
        self._xml = False
        self._fence = False

    def feed(self, piece: str, kind: str = "content") -> bool:
        """One more piece; True = this generation sprawls and should be stopped."""
        self.n += 1
        if kind == "content":
            self._scan(piece)
        if self._depth > 0 or self._xml or self._fence:
            self.active_at = self.n
        return self.limit > 0 and self.n > self.limit and self.n - self.active_at > self.quiet

    def _scan(self, piece: str) -> None:
        for ch in piece:
            if self._depth > 0:                                  # inside a JSON call: strings may contain braces and fences
                if self._in_str:
                    if self._esc:
                        self._esc = False
                    elif ch == "\\":
                        self._esc = True
                    elif ch == '"':
                        self._in_str = False
                elif ch == '"':
                    self._in_str = True
                elif ch == "{":
                    self._depth += 1
                elif ch == "}":
                    self._depth -= 1
                    if self._depth == 0:
                        self.active_at = self.n
                        self._tail = ""
                continue
            self._tail = (self._tail + ch)[-_WINDOW:]
            if self._xml:
                if ch in "<>" and _XML_END.search(self._tail):
                    self._xml, self._tail = False, ""
                    self.active_at = self.n
                continue
            if ch == '"' and _CALL_START.search(self._tail):    # `{"tool"` has just been completed: the object is open at depth 1
                self._depth, self._in_str, self._esc, self._tail = 1, False, False, ""
            elif ch in ">=" and _XML_START.search(self._tail):
                self._xml, self._tail = True, ""
            elif ch == "`" and self._tail.endswith("```"):
                self._fence = not self._fence
                self._tail = ""
                if not self._fence:
                    self.active_at = self.n
