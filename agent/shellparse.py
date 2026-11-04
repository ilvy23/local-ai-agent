"""A small, conservative bash lexer for risk analysis (agent.safety) - it never runs anything.

`split(command)` cuts a command line into the simple commands bash would run one after another or side by side, the way bash
itself reads the line:

  * separators (outside quotes, groups and substitutions): `;` `;;` `;&` `;;&` `&&` `||` `|` `|&` `&` and newline;
  * `&>` `&>>` `>&2` `2>&1` `<&0` `>|` `<>` are redirects, not separators;
  * a backslash escapes the next character; backslash-newline joins two lines;
  * `#` at the start of a word starts a comment (to the end of the line);
  * '...' is literal, "..." honours backslash escapes and may hold `$(...)` / `${...}` / backquotes, $'...' is ANSI-C quoting;
  * `( ... )`, `$( ... )`, `$(( ... ))`, `<( ... )`, `>( ... )`, backquotes and `{ ...; }` groups are kept whole (bash only
    treats `{` as a group when it starts a command and is followed by a blank, so `{a,b}` and `echo {` are plain words);
  * here-document bodies (`<<WORD`, `<<-WORD`, quoted or not; not `<<<`) are taken OUT of the command text and returned with the
    segment that owns them. `Heredoc.expands` is True for an unquoted delimiter: `$(...)` and backquotes in such a body run.

None means the line is malformed (an unterminated quote or group, a stray `)`, a `(` bash would reject, a here-document without
a delimiter) or too deeply nested - the caller must treat that as dangerous (fail closed)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Heredoc", "Segment", "split", "group_span"]


@dataclass(frozen=True)
class Heredoc:
    body: str
    expands: bool          # unquoted delimiter: substitutions in the body are executed


@dataclass
class Segment:
    text: str                                   # the simple command, without here-document bodies, comments or line continuations
    heredocs: list[Heredoc] = field(default_factory=list)


class _Bad(Exception):
    """The line cannot be analysed safely."""


# Words after which bash still expects a command (so `{` opens a group and `}` may close one there).
_COMMAND_PREFIX_WORDS = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "time"})
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?\+?=.*", re.S)
_FUNC_NAME = re.compile(r"[A-Za-z_:][\w.:-]*")
_DELIM_STOP = frozenset(" \t\n;&|<>()")
_BLANK = frozenset(" \t\n")


class _Scanner:
    def __init__(self, text: str) -> None:
        self.s = text
        self.n = len(text)

    # ── quoted spans: each returns the index just after the span ─────────────────────────────────────────────────────────
    def single(self, i: int) -> int:
        j = self.s.find("'", i)
        if j < 0:
            raise _Bad("unterminated single quote")
        return j + 1

    def ansi(self, i: int) -> int:
        s, n = self.s, self.n
        while i < n:
            if s[i] == "\\":
                i += 2
            elif s[i] == "'":
                return i + 1
            else:
                i += 1
        raise _Bad("unterminated $'...'")

    def double(self, i: int) -> int:
        s, n = self.s, self.n
        while i < n:
            c = s[i]
            nxt = s[i + 1] if i + 1 < n else ""
            if c == "\\":
                i += 2
            elif c == '"':
                return i + 1
            elif c == "$" and nxt == "(":
                i = self.paren(i + 2)
            elif c == "$" and nxt == "{":
                i = self.param(i + 2)
            elif c == "`":
                i = self.backtick(i + 1)
            else:
                i += 1
        raise _Bad("unterminated double quote")

    def backtick(self, i: int) -> int:
        s, n = self.s, self.n
        while i < n:
            if s[i] == "\\":
                i += 2
            elif s[i] == "`":
                return i + 1
            else:
                i += 1
        raise _Bad("unterminated backquote")

    def param(self, i: int) -> int:
        """`${...}`: up to the matching `}` (quotes and nested expansions honoured)."""
        s, n = self.s, self.n
        while i < n:
            c = s[i]
            nxt = s[i + 1] if i + 1 < n else ""
            if c == "\\":
                i += 2
            elif c == "'":
                i = self.single(i + 1)
            elif c == '"':
                i = self.double(i + 1)
            elif c == "`":
                i = self.backtick(i + 1)
            elif c == "$" and nxt == "(":
                i = self.paren(i + 2)
            elif c == "$" and nxt == "{":
                i = self.param(i + 2)
            elif c == "}":
                return i + 1
            else:
                i += 1
        raise _Bad("unterminated ${...}")

    def paren(self, i: int) -> int:
        """A command list up to the matching `)` (for `(`, `$(`, `<(`, `>(`)."""
        return self.commands(i, ")", collect=False)[1]

    def group(self, i: int) -> int:
        """A command list up to the `}` that closes a `{ ...` group."""
        return self.commands(i, "}", collect=False)[1]

    def delimiter(self, k: int) -> tuple[int, str, bool]:
        """The here-document delimiter word starting at `k`: (index after it, the delimiter, was any part quoted)."""
        s, n = self.s, self.n
        while k < n and s[k] in " \t":
            k += 1
        start, out, quoted = k, [], False
        while k < n and s[k] not in _DELIM_STOP:
            c = s[k]
            if c == "'":
                j = self.single(k + 1)
                out.append(s[k + 1:j - 1])
                quoted, k = True, j
            elif c == '"':
                j = self.double(k + 1)
                out.append(re.sub(r'\\(["\\$`])', r"\1", s[k + 1:j - 1]))
                quoted, k = True, j
            elif c == "\\" and k + 1 < n:
                out.append(s[k + 1])
                quoted, k = True, k + 2
            else:
                out.append(c)
                k += 1
        if k == start:
            raise _Bad("here-document without a delimiter")
        return k, "".join(out), quoted

    def bodies(self, i: int, pending: list[tuple[str, bool, bool, list[Heredoc]]]) -> int:
        """Read the bodies of the here-documents announced on the line that just ended; returns the index after the last one."""
        s, n = self.s, self.n
        for delim, strip_tabs, expands, target in pending:
            lines: list[str] = []
            while i < n:
                j = s.find("\n", i)
                end = n if j < 0 else j
                line = s[i:end]
                i = end + 1 if j >= 0 else n
                if (line.lstrip("\t") if strip_tabs else line) == delim:
                    break
                lines.append(line)
            target.append(Heredoc("\n".join(lines), expands))
        return i

    # ── the command-list scanner ───────────────────────────────────────────────────────────────────────────────────────────
    def commands(self, i: int, closer: str | None, collect: bool) -> tuple[list[Segment], int]:
        s, n = self.s, self.n
        segs: list[Segment] = []
        buf: list[str] = []
        word: list[str] = []
        docs: list[Heredoc] = []
        pending: list[tuple[str, bool, bool, list[Heredoc]]] = []
        cmd_pos = True                       # bash expects a command word here (reserved words, `{`, `}` count only there)
        func_name_next = False               # after `function`: the name, then the body (a `{ ...; }` group) follows

        def end_word() -> None:
            nonlocal cmd_pos, func_name_next
            w = "".join(word)
            if w:
                if func_name_next:
                    func_name_next = False   # `function NAME` - the body comes next, at a command position
                elif cmd_pos and w == "function":
                    func_name_next = True
                elif not (w in _COMMAND_PREFIX_WORDS or _ASSIGNMENT.fullmatch(w)):
                    cmd_pos = False
            word.clear()

        def flush() -> None:
            nonlocal docs, cmd_pos
            end_word()
            if collect:
                text = "".join(buf).strip()
                if text or docs:
                    segs.append(Segment(text, docs))
            buf.clear()
            docs = []
            cmd_pos = True

        def separator(k: int) -> None:
            nonlocal i, cmd_pos
            if collect:
                flush()
            else:
                end_word()
                buf.append(s[i:i + k])
                cmd_pos = True
            i += k

        def keep(j: int) -> None:
            nonlocal i
            buf.append(s[i:j])
            word.append(s[i:j])
            i = j

        while i < n:
            c = s[i]
            nxt = s[i + 1] if i + 1 < n else ""
            if c == "\\":
                if nxt == "\n":                       # line continuation: the two lines are one
                    i += 2
                    continue
                if not nxt:
                    raise _Bad("trailing backslash")
                keep(i + 2)
                continue
            if c == "\n":
                end_word()
                i += 1
                if pending:
                    i = self.bodies(i, pending)
                    pending.clear()
                if collect:
                    flush()
                else:
                    buf.append("\n")
                    cmd_pos = True
                continue
            if c in " \t\r":
                end_word()
                buf.append(c)
                i += 1
                continue
            if c == "#" and not word:                 # a comment, up to (not including) the newline
                j = s.find("\n", i)
                i = n if j < 0 else j
                continue
            if c == "'":
                keep(self.single(i + 1))
                continue
            if c == '"':
                keep(self.double(i + 1))
                continue
            if c == "`":
                keep(self.backtick(i + 1))
                continue
            if c == "$" and nxt == "'":
                keep(self.ansi(i + 2))
                continue
            if c == "$" and nxt == "(":
                keep(self.paren(i + 2))
                continue
            if c == "$" and nxt == "{":
                keep(self.param(i + 2))
                continue
            if c in "<>" and nxt == "(":              # process substitution
                end_word()
                keep(self.paren(i + 2))
                continue
            if c == "(":
                w = "".join(word)
                after = s[i + 1:].lstrip(" \t")
                if w and after.startswith(")") and _FUNC_NAME.fullmatch(w):      # `name()` - a function definition
                    keep(s.index(")", i + 1) + 1)
                    word.clear()
                    cmd_pos = True
                    continue
                if (cmd_pos and not w) or w.endswith("=") or (w and w[-1] in "?*+@!"):   # subshell / ((...)) / array / extglob
                    keep(self.paren(i + 1))
                    continue
                raise _Bad("unexpected '('")
            if c == ")":
                if closer == ")":
                    end_word()
                    return segs, i + 1
                raise _Bad("unbalanced ')'")
            if c == "{" and cmd_pos and not word and (not nxt or nxt in _BLANK):
                keep(self.group(i + 1))
                end_word()
                cmd_pos = False
                continue
            if c == "}" and closer == "}" and cmd_pos and not word:
                return segs, i + 1
            if c == "<":
                if s.startswith("<<<", i):
                    end_word()
                    buf.append("<<<")
                    i += 3
                    continue
                if nxt == "<":
                    strip_tabs = s.startswith("<<-", i)
                    k, delim, quoted = self.delimiter(i + (3 if strip_tabs else 2))
                    pending.append((delim, strip_tabs, not quoted, docs))
                    end_word()
                    buf.append(s[i:k])
                    i = k
                    continue
                op = 2 if nxt in "&>" else 1
                end_word()
                buf.append(s[i:i + op])
                i += op
                continue
            if c == ">":
                op = 2 if nxt in ">|&" else 1
                end_word()
                buf.append(s[i:i + op])
                i += op
                continue
            if c == "&":
                if nxt == "&":
                    separator(2)
                elif nxt == ">":
                    op = 3 if s.startswith("&>>", i) else 2
                    end_word()
                    buf.append(s[i:i + op])
                    i += op
                else:
                    separator(1)                       # `cmd &`: runs in the background, the next command starts at once
                continue
            if c == "|":
                separator(2 if nxt in "|&" else 1)
                continue
            if c == ";":
                separator(3 if s.startswith(";;&", i) else 2 if nxt in ";&" else 1)
                continue
            buf.append(c)
            word.append(c)
            i += 1
        if closer is not None:
            raise _Bad(f"missing '{closer}'")
        if pending:                                     # `cat <<EOF` with no body at all: bash reads an empty document
            self.bodies(n, pending)
        flush()
        return segs, n


def split(command: str) -> list[Segment] | None:
    """The simple commands of `command` (see the module docstring), or None when it cannot be analysed safely."""
    try:
        return _Scanner(command or "").commands(0, None, collect=True)[0]
    except (_Bad, RecursionError, IndexError, ValueError):
        return None


def group_span(text: str) -> tuple[str, str] | None:
    """For a segment that starts with a `( ... )` subshell or a `{ ...; }` group: (the inner command list, what follows the group,
    e.g. a redirect). None when `text` does not start with a group, or the group is malformed."""
    t = text.lstrip()
    if not t:
        return None
    try:
        sc = _Scanner(t)
        if t[0] == "(":
            end = sc.paren(1)
        elif t[0] == "{" and len(t) > 1 and t[1] in _BLANK:
            end = sc.group(1)
        else:
            return None
    except (_Bad, RecursionError, IndexError, ValueError):
        return None
    inner = t[1:end - 1].strip()
    if t[0] == "{":
        inner = inner.rstrip(";").strip()
    return inner, t[end:]
