from __future__ import annotations

import companion.menu as m
from companion.menu import _fill


class StubConsole:
    def __init__(self, answers):
        self.answers = list(answers)

    def input(self, _prompt=""):
        return self.answers.pop(0)

    def print(self, *_a, **_k):
        pass


def test_fill_prompts_for_placeholder():
    # A {setting}/{value} placeholder prompts the user and substitutes the answer.
    argv = _fill(["settings", "set", "{setting}", "{value}"], StubConsole(["model", "gemma2:9b"]))
    assert argv == ["settings", "set", "model", "gemma2:9b"]


def test_fill_blank_placeholder_cancels():
    assert _fill(["settings", "set", "{setting}", "{value}"], StubConsole([""])) is None


def test_fill_passes_literals_through():
    argv = _fill(["node", "display", "off"], StubConsole([]))
    assert argv == ["node", "display", "off"]


def test_pick_from_number_and_freetext():
    rows = [("First option", "alpha"), ("Second option", "beta")]
    assert m._pick_from(StubConsole(["2"]), rows, "choice") == "beta"
    assert m._pick_from(StubConsole(["typed"]), rows, "choice") == "typed"  # free text
    assert m._pick_from(StubConsole([""]), rows, "contact") is None
