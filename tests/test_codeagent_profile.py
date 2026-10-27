from __future__ import annotations

import json

from agent.codeagent import profile as PR


def test_python_project_with_tests_and_scripts(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "probe_python", lambda: {"version": "3.13.1", "pytest": False, "uv": True})
    (tmp_path / "main.py").write_text("print(1)\n")
    (tmp_path / "test_main.py").write_text("def test_a():\n    assert True\n")
    p = PR.detect(tmp_path)
    assert "python" in p.kinds and p.run_hint == "python3 main.py"
    assert p.test_cmd.startswith("uv run --no-project --with pytest")          # system python has no pytest -> go through uv
    assert any("NOT installed" in n and "never pip install" in n for n in p.notes)
    assert "tests: uv run" in p.text()
    monkeypatch.setattr(PR, "probe_python", lambda: {"version": "3.13.1", "pytest": True, "uv": True})
    assert PR.detect(tmp_path).test_cmd == "python3 -m pytest -q -x"


def test_node_rust_make_and_override(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "probe_python", lambda: {"version": "", "pytest": False, "uv": False})
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "jest", "build": "tsc", "dev": "vite"}}))
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    (tmp_path / "Makefile").write_text("lint:\n\techo lint\n")
    p = PR.detect(tmp_path)
    assert {"node", "rust", "make"} <= set(p.kinds)
    assert p.test_cmd == "npm test --silent" and p.build_cmd == "npm run build" and p.run_hint == "npm run dev"
    assert p.lint_cmd == "make lint" or p.lint_cmd == "cargo clippy -q"
    assert PR.detect(tmp_path, override_test="just check").test_cmd == "just check"


def test_check_commands_always_compile_python(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "probe_python", lambda: {"version": "3.13", "pytest": True, "uv": False})
    (tmp_path / "game.py").write_text("x = 1\n")
    p = PR.detect(tmp_path)
    cmds = PR.check_commands(p, ["game.py", "notes.md"])
    assert cmds[0].startswith("python3 -m py_compile") and "game.py" in cmds[0] and "notes.md" not in cmds[0]
    assert PR.check_commands(p, ["notes.md"]) == ([p.test_cmd] if p.test_cmd else [])


def test_check_commands_quote_odd_file_names_for_the_shell(tmp_path):
    import subprocess

    odd = "it's $HOME `x` grüße.py"
    (tmp_path / odd).write_text("x = 1\n")
    cmd = PR.check_commands(PR.Profile(tmp_path), [odd])[0]
    r = subprocess.run(["bash", "-c", cmd], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
