from agent.coding.repomap import extract_tags, rank_files, render_map


def test_extract_defs_and_refs():
    tags = extract_tags("m.py", "def foo():\n    return bar()\n")
    assert "foo" in tags.defines
    assert tags.references["bar"] == 1


def test_extract_ignores_own_defs_as_refs():
    tags = extract_tags("m.py", "def foo():\n    return foo()\n")
    assert "foo" not in tags.references


def test_extract_handles_syntax_error():
    tags = extract_tags("m.py", "def broken(:\n")
    assert tags.defines == frozenset()


def test_rank_prefers_widely_referenced_file():
    files = {
        "core.py": "def shared():\n    return 1\n",
        "a.py": "from core import shared\n\ndef a():\n    return shared()\n",
        "b.py": "from core import shared\n\ndef b():\n    return shared()\n",
    }
    ranked = rank_files(files)
    assert ranked[0].path == "core.py"


def test_chat_files_boost_rank():
    files = {
        "core.py": "def shared():\n    return 1\n",
        "leaf.py": "x = 1\n",
    }
    ranked = rank_files(files, chat_files={"leaf.py"})
    top = {r.path: r.rank for r in ranked}
    assert top["leaf.py"] > 0


def test_render_map_respects_budget():
    files = {f"m{i}.py": f"def f{i}():\n    return {i}\n" for i in range(30)}
    out = render_map(files, budget_tokens=20)
    assert len(out) < 400


def test_render_map_excludes_chat_files():
    files = {
        "core.py": "def shared():\n    return 1\n",
        "a.py": "from core import shared\n\ndef a():\n    return shared()\n",
    }
    out = render_map(files, chat_files={"core.py"})
    assert "core.py:" not in out


def test_empty_files_gives_empty():
    assert rank_files({}) == []
