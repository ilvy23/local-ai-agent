"""The persistent project map (agent/codeagent/projectmap.py): incremental refresh, renames carried into the notes, stale marking, where it is stored, and its use in the engine."""
from __future__ import annotations

import os

import pytest

from agent.codeagent import projectmap as P

CORE = '"""core"""\nDEFAULT = 1\n\ndef run(a, b):\n    """Run the thing."""\n    return a + b\n\nclass Box:\n    """A box."""\n    def open(self): ...\n'


def _proj(tmp_path):
    (tmp_path / "core.py").write_text(CORE)
    (tmp_path / "app.py").write_text("import core\n\ndef main():\n    return core.run(1, 2)\n")
    return tmp_path


def test_refresh_is_incremental_and_reads_unchanged_files_from_the_cache(tmp_path, monkeypatch):
    root = _proj(tmp_path)
    store = tmp_path / "store"
    first = P.refresh(root, store)
    assert sorted(first.changed) == ["app.py", "core.py"] and "run(a, b) - Run the thing." in (store / "MAP.md").read_text()
    reads = []
    real = type(root).read_bytes
    monkeypatch.setattr(type(root), "read_bytes", lambda self: (reads.append(self.name), real(self))[1])
    assert P.refresh(root, store).changed == [] and reads == []                               # nothing changed: a stat per file, no reads
    (root / "app.py").write_text("import core\n\ndef main():\n    return 3\n")
    again = P.refresh(root, store)
    assert again.changed == ["app.py"] and reads == ["app.py"]


def test_a_renamed_function_is_updated_in_the_notes_and_a_deleted_one_is_marked_stale(tmp_path):
    root = _proj(tmp_path)
    store = tmp_path / "store"
    P.refresh(root, store)
    f = store / "MAP.md"
    f.write_text(f.read_text() + "\n## Notes\n- `run` is called by main(); Box.open does nothing yet; DEFAULT is the seed\n")
    (root / "core.py").write_text(CORE.replace("def run(a, b):", "def execute(a, b):"))                       # same signature and doc: a rename
    ch = P.refresh(root, store)
    text = f.read_text()
    assert ch.renamed_names == {"run": "execute"} and "`execute` is called by main()" in text and "`run`" not in text
    (root / "core.py").write_text(CORE.replace("def run(a, b):", "def execute(a, b):").replace("DEFAULT = 1\n", ""))
    ch = P.refresh(root, store)
    assert "DEFAULT" in ch.stale and "DEFAULT is the seed [stale: `DEFAULT` is gone]" in f.read_text()
    P.refresh(root, store)
    assert f.read_text().count("[stale:") == 1                                                      # marked once, not on every refresh


def test_a_moved_file_updates_the_path_in_the_notes(tmp_path):
    root = _proj(tmp_path)
    store = tmp_path / "store"
    P.refresh(root, store)
    f = store / "MAP.md"
    f.write_text(f.read_text() + "\n## Notes\n- core.py holds the engine; see core.py:run\n")
    (root / "lib").mkdir()
    os.replace(root / "core.py", root / "lib" / "core.py")
    ch = P.refresh(root, store)
    assert ch.renamed_files == {"core.py": "lib/core.py"}
    assert "- lib/core.py holds the engine; see lib/core.py:run" in f.read_text()
    (root / "lib" / "core.py").unlink()
    ch = P.refresh(root, store)
    assert "lib/core.py" in ch.stale


def test_renamed_names_pairs_by_signature_or_by_being_the_only_pair():
    old = ["run(a, b) - Run it.", "helper(x)", "class Box [open] - A box."]
    assert P.renamed_names(old, ["execute(a, b) - Run it.", "helper(x)", "class Box [open] - A box."]) == {"run": "execute"}
    assert P.renamed_names(["only(a)"], ["other(a, b, c)"]) == {"only": "other"}
    assert P.renamed_names(["a(x)", "b(y)"], ["c(p)", "d(q)"]) == {}                                 # two unrelated swaps: no guess
    assert P.renamed_names(["Old"], ["New"]) == {"Old": "New"} and P.renamed_names(["x(a)"], ["x(a)"]) == {}


def test_the_store_is_dot_night_when_it_exists_else_the_state_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    root = tmp_path / "proj"
    root.mkdir()
    s1 = P.store_dir(root)
    assert str(s1).startswith(str(tmp_path / "state" / "agent" / "maps")) and s1 == P.store_dir(root) and s1 != P.store_dir(tmp_path / "other")
    (root / ".night").mkdir()
    assert P.store_dir(root) == root.resolve() / ".night"
    assert not (root / ".night" / "MAP.md").exists()


def test_text_for_the_prompt_respects_the_budget_and_skips_projects_without_code(tmp_path):
    root = _proj(tmp_path)
    store = tmp_path / "store"
    P.refresh(root, store)
    (store / "MAP.md").write_text((store / "MAP.md").read_text() + "\n## Notes\n" + "note line\n" * 500)
    body = P.text(root, budget=1500, store=store, notes_budget=400)
    assert len(body) <= 1500 + 410 and "core.py" in body and "## Notes" in body and body.count("note line") < 60
    empty = tmp_path / "empty"
    empty.mkdir()
    P.refresh(empty, tmp_path / "s2")
    assert P.text(empty, store=tmp_path / "s2") == ""
    assert P.text(tmp_path / "nope", store=tmp_path / "nowhere") == ""


def test_refresh_safe_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "refresh", lambda *a, **k: 1 / 0)
    assert P.refresh_safe(tmp_path) is None


def test_the_engine_puts_the_persistent_map_into_its_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from tests.test_codeagent_engine import FakeLLM, engine

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "wren.py").write_text('def bake(n):\n    """Bake n rolls."""\n    return n\n')
    eng.start()
    head = "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "system")
    assert "PROJECT MAP" in head and "wren.py" in head and "bake(n) - Bake n rolls." in head and "do not read files only to learn what is where" in head


def test_a_project_without_source_files_still_gets_the_plain_repo_map(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from tests.test_codeagent_engine import FakeLLM, engine

    eng, repo, _ = engine(tmp_path, FakeLLM([]))
    (repo / "notes.txt").write_text("hello\n")
    eng.start()
    head = "\n".join(m["content"] for m in eng.ctx.messages if m["role"] == "system")
    assert "PROJECT MAP" not in head and "REPO MAP" in head
