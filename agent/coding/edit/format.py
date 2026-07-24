from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class EditKind(str, Enum):
    WHOLE_FILE = "whole_file"
    SEARCH_REPLACE = "search_replace"


@dataclass(frozen=True)
class Edit:
    path: str
    kind: EditKind
    replace: str
    search: str | None = None


@dataclass(frozen=True)
class ParseResult:
    edits: tuple[Edit, ...]
    malformed: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.edits) and not self.malformed


_BLOCK_RE = re.compile(
    r"^<<<<[ \t]+FILE:[ \t]*(?P<path>.+?)[ \t]*\n(?P<body>.*?)^>>>>[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_SEARCH_MARKER = re.compile(r"^------[ \t]*SEARCH[ \t]*$", re.MULTILINE)
_REPLACE_MARKER = re.compile(r"^======[ \t]*REPLACE[ \t]*$", re.MULTILINE)


def _strip_one_trailing_newline(text: str) -> str:
    return text.removesuffix("\n")


def parse_edits(response: str) -> ParseResult:
    blocks = list(_BLOCK_RE.finditer(response))
    if not blocks:
        if "<<<<" in response and "FILE:" in response:
            reason = "an edit block was opened with '<<<< FILE:' but never closed with '>>>>'"
        else:
            reason = "no edit blocks found; expected at least one '<<<< FILE: <path>' block"
        return ParseResult(edits=(), malformed=(reason,))

    edits: list[Edit] = []
    malformed: list[str] = []

    for block in blocks:
        path = block.group("path").strip()
        body = block.group("body")
        edit, reason = _parse_block(path, body)
        if edit is not None:
            edits.append(edit)
        else:
            malformed.append(reason)

    return ParseResult(edits=tuple(edits), malformed=tuple(malformed))


def _parse_block(path: str, body: str) -> tuple[Edit | None, str]:
    if not path:
        return None, "an edit block is missing its file path after 'FILE:'"

    search_marker = _SEARCH_MARKER.search(body)
    replace_marker = _REPLACE_MARKER.search(body)

    if search_marker is None and replace_marker is None:
        return (
            Edit(
                path=path,
                kind=EditKind.WHOLE_FILE,
                replace=_strip_one_trailing_newline(body),
            ),
            "",
        )

    if search_marker is None or replace_marker is None:
        missing = "'------ SEARCH'" if search_marker is None else "'====== REPLACE'"
        return None, f"search/replace block for {path} is missing its {missing} marker"

    if search_marker.start() > replace_marker.start():
        return None, f"search/replace block for {path} has REPLACE before SEARCH"

    search = _strip_one_trailing_newline(body[search_marker.end() + 1 : replace_marker.start()])
    replace = _strip_one_trailing_newline(body[replace_marker.end() + 1 :])
    if not search:
        return None, f"search/replace block for {path} has an empty SEARCH section"

    return (
        Edit(path=path, kind=EditKind.SEARCH_REPLACE, replace=replace, search=search),
        "",
    )
