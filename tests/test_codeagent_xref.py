from __future__ import annotations

from agent.codeagent import xref as X

ENGINE = '''
from dataclasses import dataclass

CONSTANT = 3

class State:
    def __init__(self, hp=10, name="x"):
        self.hp = hp
        self.items = []

    def take(self, item):
        self.items.append(item)

    @property
    def alive(self):
        return self.hp > 0

@dataclass
class Item:
    name: str
    weight: int = 1

def save(state, path):
    pass

class Child(State):
    def extra(self):
        pass

class Weird:
    def __getattr__(self, name):
        return 1
'''


def project(tmp_path, main_src):
    (tmp_path / "engine.py").write_text(ENGINE)
    (tmp_path / "main.py").write_text(main_src)
    return X.check_file(tmp_path, tmp_path / "main.py", main_src)


def test_invented_method_and_wrong_import_are_found(tmp_path):
    w = project(tmp_path, "from engine import State, load_game\nimport engine\ns = State()\ns.gain_item('key')\nprint(engine.nothing)\n")
    text = "\n".join(w)
    assert "does not define 'load_game'" in text
    assert "`State` (from engine.py) has no attribute 'gain_item'" in text and "take" in text
    assert "defines no 'nothing'" in text


def test_correct_code_has_no_warnings(tmp_path):
    src = ("import engine\nfrom engine import State, Item, save, Child\n"
           "s = State(5)\ns.take(Item('a'))\nprint(s.hp, s.items, s.alive, engine.CONSTANT)\nsave(s, 'x.json')\n"
           "c = Child()\nc.extra(); c.take(1); print(c.hp)\nengine.save(s, 'p')\n")
    assert project(tmp_path, src) == []


def test_argument_mismatches(tmp_path):
    w = project(tmp_path, "from engine import State, Item, save\nState(1, 2, 3)\nsave(State())\nItem()\nItem('a', 1, 2)\nState(color='red')\n")
    text = "\n".join(w)
    assert "takes at most 2 positional" in text and "missing required argument(s): path" in text
    assert "missing required argument(s): name" in text and "has no parameter named 'color'" in text


def test_self_attribute_types_and_open_classes_are_respected(tmp_path):
    src = ("from engine import State, Weird\nclass Game:\n    def __init__(self):\n        self.state = State()\n"
           "        self.w = Weird()\n    def go(self):\n        self.state.take(1)\n        self.state.gain_item(2)\n        print(self.w.anything)\n")
    w = project(tmp_path, src)
    assert len(w) == 1 and "gain_item" in w[0]                          # Weird has __getattr__ -> never flagged


def test_note_formatting_and_unknown_files(tmp_path):
    (tmp_path / "engine.py").write_text(ENGINE)
    (tmp_path / "main.py").write_text("from engine import State\nState().bogus()\n")
    n = X.note(tmp_path, tmp_path / "main.py", (tmp_path / "main.py").read_text())
    assert n.startswith("\nXREF") and "bogus" in n
    assert X.note(tmp_path, tmp_path / "notes.md", "text") == "" and X.check_file(tmp_path, tmp_path / "main.py", "def (:") == []
