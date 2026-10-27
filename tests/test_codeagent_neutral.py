"""Language / project / model neutrality of the coding agent: C1 syntax checks for non-Python files (missing tool = skipped
silently), C2 neutral prompt wording and scratch handling, C3 the narrow llm seam."""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys

import pytest

from agent.codeagent import edits as E, profile as PR, static, symbols, xref
from agent.codeagent.context import ContextManager
from agent.codeagent.eventlog import EventLog
from agent.codeagent.tools import CodeTools


# ── C1 ──────────────────────────────────────────────────────────────────────

def fake_tool(bindir, name, *, code=0, out="", err=""):
    """A stand-in compiler on PATH: swallows stdin, prints, exits."""
    f = bindir / name
    f.write_text(f"#!/bin/sh\ncat >/dev/null 2>&1\nprintf '%s' '{out}'\nprintf '%s' '{err}' >&2\nexit {code}\n")
    f.chmod(f.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def bare_path(tmp_path, monkeypatch):
    """PATH with only a directory of our fakes (and sh for them): every real checker is 'missing'."""
    d = tmp_path / "bin"
    d.mkdir()
    (d / "sh").symlink_to("/bin/sh")
    for tool in ("cat", "grep"):
        (d / tool).symlink_to(shutil.which(tool))
    monkeypatch.setenv("PATH", str(d))
    return d


BROKEN = {"a.ts": "const x: = ;", "a.tsx": "const x = <", "a.mjs": "const = ;", "a.rs": "fn main( {", "a.go": "package", "a.c": "int main( {",
          "a.cpp": "int main( {", "a.cc": "int f( {", "a.java": "class A {"}


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_missing_tool_is_skipped_silently(bare_path, name):
    assert E.check_syntax(name, BROKEN[name]) is None


def test_missing_bash_and_node_are_skipped_silently(bare_path):
    assert E.check_syntax("a.sh", "if then fi (") is None
    assert E.check_syntax("a.js", "const = ;") is None


def test_go_rust_c_ts_with_tools_present(bare_path):
    fake_tool(bare_path, "gofmt", code=2, err="<standard input>:1:1: expected package, found EOF")
    msg = E.check_syntax("a.go", "x")
    assert msg and "expected package" in msg and "a.go" in msg
    fake_tool(bare_path, "gofmt", code=0, out="package main")
    assert E.check_syntax("a.go", "package main\n") is None

    fake_tool(bare_path, "rustfmt", code=1, err="error: expected one of `!` or `::`, found `}`")
    msg = E.check_syntax("a.rs", "fn main( {")
    assert msg and "expected one of" in msg
    fake_tool(bare_path, "rustfmt", code=1, out="Diff in <stdin> at line 1:")          # only formatting differs: not a syntax error
    assert E.check_syntax("a.rs", "fn main(){}") is None

    fake_tool(bare_path, "cc", code=1, err="<stdin>:1:11: error: expected declaration specifiers")
    assert "expected declaration" in (E.check_syntax("a.c", "int main( {") or "")
    fake_tool(bare_path, "cc", code=1, err="<stdin>:1:10: fatal error: foo.h: No such file or directory")   # cannot judge without the header
    assert E.check_syntax("a.c", '#include "foo.h"\n') is None
    fake_tool(bare_path, "c++", code=1, err="<stdin>:1:1: error: expected unqualified-id")
    assert "expected unqualified-id" in (E.check_syntax("a.cpp", "}") or "")

    fake_tool(bare_path, "tsc", code=2, out="a.ts(1,10): error TS1005: ';' expected.\na.ts(2,1): error TS2307: Cannot find module 'x'.")
    msg = E.check_syntax("a.ts", "const x: = ;")
    assert msg and "TS1005" in msg and "TS2307" not in msg                               # only syntax diagnostics, never type noise
    fake_tool(bare_path, "tsc", code=2, out="a.ts(2,1): error TS2307: Cannot find module 'x'.")
    assert E.check_syntax("a.ts", "import x from 'x'") is None


def test_real_json_toml_and_bash():
    assert E.check_syntax("a.json", '{"a": }') and E.check_syntax("a.json", "{}") is None
    assert E.check_syntax("a.toml", "a = ") and E.check_syntax("a.toml", "a = 1\n") is None
    if not shutil.which("bash"):
        pytest.skip("no bash")
    assert "Shell syntax error" in (E.check_syntax("run.sh", "if true; then\n") or "")
    assert E.check_syntax("run.sh", "echo hi\n") is None


def test_yaml_without_pyyaml_is_skipped_silently(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)                                       # `import yaml` raises ImportError
    assert E.check_syntax("a.yaml", "a: [1, 2") is None


def test_check_cmd_overrides_per_project(tmp_path, bare_path):
    cmd = "grep -q GOOD {file}"
    assert E.check_syntax("a.ts", "nope", check_cmd=cmd) and "a.ts" in E.check_syntax("a.ts", "nope", check_cmd=cmd)
    assert E.check_syntax("a.ts", "GOOD", check_cmd=cmd) is None
    assert E.check_syntax("a.kt", "nope", check_cmd=cmd) is not None                    # languages without a built-in checker too
    assert E.check_syntax("notes.md", "nope", check_cmd=cmd) is None                     # prose / data stay unchecked
    assert E.check_syntax("a.py", "x = 1\n", check_cmd=cmd) is None                      # Python keeps its own gate
    assert E.check_syntax("a.ts", "nope", check_cmd="no-such-tool-xyz {file}") is None   # missing tool: silent


def test_check_commands_cover_other_languages(tmp_path, bare_path):
    for t in ("node", "bash", "gofmt"):
        fake_tool(bare_path, t)
    p = PR.Profile(tmp_path)
    cmds = PR.check_commands(p, ["a.js", "b.sh", "c.go", "d.txt"])
    joined = " && ".join(cmds)
    assert "node --check a.js" in joined and "bash -n b.sh" in joined and "gofmt -e" in joined and "d.txt" not in joined
    fake_tool(bare_path, "eslint")
    assert PR.check_commands(p, ["a.js"], check_cmd="eslint {file}") == ["eslint a.js"]


def test_check_commands_skip_missing_tools(tmp_path, bare_path):
    assert PR.check_commands(PR.Profile(tmp_path), ["a.js", "b.sh", "c.go"]) == []


def test_static_xref_skip_other_file_types(tmp_path):
    assert static.check(tmp_path, tmp_path / "a.ts", "const x = (") == []
    assert static.check_files(tmp_path, ["a.ts"]) == {}
    assert xref.check_file(tmp_path, tmp_path / "a.go", "package main\nfunc f( {") == []
    assert xref.note(tmp_path, tmp_path / "a.rs", "fn main( {") == ""


def _tools(tmp_path):
    ctx = ContextManager(None, {"models": {"background": "m"}}, 32768, EventLog("t", tmp_path / "e.jsonl"), tmp_path)
    return CodeTools(tmp_path, PR.detect(tmp_path), ctx)


def test_show_symbol_without_path_finds_non_python_definitions(tmp_path):
    (tmp_path / "util.go").write_text("package main\n\nfunc (s *Server) Handle(w int) int {\n\treturn w\n}\n\nfunc other() {}\n")
    (tmp_path / "app.ts").write_text("export class Widget {\n  run() {}\n}\n")
    (tmp_path / "lib.rs").write_text("pub fn compute(x: i32) -> i32 {\n    x + 1\n}\n")
    (tmp_path / "m.c").write_text("#include <stdio.h>\nint add(int a, int b);\nint add(int a, int b)\n{\n  return a + b;\n}\n")
    t = _tools(tmp_path)
    assert "lines 3-5" in t.show_symbol("Handle") and "util.go" in t.show_symbol("Handle")
    assert "app.ts" in t.show_symbol("Widget")
    assert "lib.rs" in t.show_symbol("compute")
    out = t.show_symbol("add")
    assert "m.c" in out and "return a + b" in out
    assert "no definition of 'nothing_here'" in t.show_symbol("nothing_here")


def test_show_symbol_without_path_ambiguous_across_languages(tmp_path):
    (tmp_path / "a.py").write_text("def run():\n    pass\n")
    (tmp_path / "b.js").write_text("function run() {\n}\n")
    out = _tools(tmp_path).show_symbol("run")
    assert "defined in 2 files" in out and "a.py" in out and "b.js" in out


def test_generic_span_c_and_go_forms():
    assert symbols.generic_span("int x;\nvoid foo(int a) {\n  bar();\n}\n", "foo").end == 4
    assert symbols.generic_span("int y = foo(3);\n", "foo") is None                      # a call is not a definition


# ── C2 ──────────────────────────────────────────────────────────────────────

def test_generic_system_prompt_is_project_and_os_neutral():
    from agent.codeagent.engine import SYSTEM
    assert "game" not in SYSTEM.lower() and "nixos" not in SYSTEM.lower()
    assert ".scratch" in SYSTEM and "real data files" in SYSTEM          # the rule itself stays


def _probe(readonly):
    return lambda: {"version": "3.13.1", "pytest": False, "uv": True, "readonly": readonly}


@pytest.mark.parametrize("nixos,readonly,expect", [(True, True, True), (True, False, False), (False, True, False), (False, False, False)])
def test_nixos_note_only_when_nixos_and_python_is_read_only(tmp_path, monkeypatch, nixos, readonly, expect):
    monkeypatch.setattr(PR, "probe_python", _probe(readonly))
    monkeypatch.setattr(PR, "is_nixos", lambda: nixos)
    (tmp_path / "main.py").write_text("print(1)\n")
    text = PR.detect(tmp_path).text()
    assert ("NixOS" in text) is expect
    assert "never pip install" in text                                       # the generic advice stays either way


def test_is_nixos_reads_marker_file_and_os_release(tmp_path, monkeypatch):
    monkeypatch.setattr(PR, "NIXOS_MARKER", tmp_path / "NIXOS")
    monkeypatch.setattr(PR, "OS_RELEASE", tmp_path / "os-release")
    assert PR.is_nixos() is False
    (tmp_path / "os-release").write_text('NAME="Ubuntu"\nID=ubuntu\n')
    assert PR.is_nixos() is False
    (tmp_path / "os-release").write_text('NAME=NixOS\nID=nixos\n')
    assert PR.is_nixos() is True
    (tmp_path / "os-release").write_text('ID=arch\n')
    (tmp_path / "NIXOS").write_text("")
    assert PR.is_nixos() is True


def test_scratch_is_excluded_via_git_info_exclude_once(tmp_path):
    (tmp_path / ".git" / "info").mkdir(parents=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("# comment\n*.swp")                # no trailing newline
    assert E.ensure_scratch_excluded(tmp_path) is True
    assert E.ensure_scratch_excluded(tmp_path) is False                                    # idempotent
    lines = (tmp_path / ".git" / "info" / "exclude").read_text().splitlines()
    assert lines == ["# comment", "*.swp", ".scratch/"]
    assert not (tmp_path / ".gitignore").exists()


def test_scratch_exclude_creates_missing_info_dir_and_is_noop_outside_git(tmp_path):
    repo, plain = tmp_path / "repo", tmp_path / "plain"
    (repo / ".git").mkdir(parents=True)
    plain.mkdir()
    assert E.ensure_scratch_excluded(repo) is True and (repo / ".git" / "info" / "exclude").read_text() == ".scratch/\n"
    assert E.ensure_scratch_excluded(plain) is False
    assert list(plain.iterdir()) == []


def test_scratch_exclude_really_hides_it_from_git_add_all(tmp_path):
    if not shutil.which("git"):
        pytest.skip("no git")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    t = _tools(tmp_path)
    t.run("mkdir -p .scratch && echo hi > .scratch/t.txt")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=tmp_path, capture_output=True, text=True).stdout.split()
    assert staged == ["a.py"]


def test_spill_into_scratch_excludes_it_too(tmp_path):
    (tmp_path / ".git" / "info").mkdir(parents=True)
    t = _tools(tmp_path)
    assert "full output: .scratch/spill/run-1.txt" in t._spill("x\n" * 5000, 0.1)
    assert ".scratch/" in (tmp_path / ".git" / "info" / "exclude").read_text().splitlines()


@pytest.mark.parametrize("target,scratch", [
    ("src/tmpfile.py", False), ("scratchpad_notes.py", False), ("src/scratchpad_notes.py", False), ("tmpl/x.py", False), ("tmp_utils.py", False),
    ("tmp/x.py", True), ("src/tmp/x.py", True), ("pkg/.scratch/t.py", True), ("a/scratch/b/c.py", True), ("./.tmp/x", True), ("a\\tmp\\x.py", True),
])
def test_scratch_decided_by_any_path_segment_not_prefix(target, scratch):
    from agent.codeagent.lookact import is_scratch_path
    assert is_scratch_path(target) is scratch


# ── C3 ──────────────────────────────────────────────────────────────────────

class MinimalLLM:
    """Everything an OpenAI-compatible backend must offer: no server_url, revive, loaded_models, unload, unload_all, set_read_timeout,
    chat_with_tools_stream, no /api/ps, no keep_alive."""

    def __init__(self, replies=()):
        self.replies = list(replies)

    def chat_with_tools(self, messages, model, tools, num_ctx=None, think=None, extra_options=None):
        return dict(self.replies.pop(0))

    def chat(self, messages, model=None, stream=False, background=False, **kw):
        yield "NONE"


def test_llm_seam_helpers_default_when_the_backend_lacks_the_methods():
    from agent import llm as L
    m = MinimalLLM()
    assert L.server_url_of(m, "x") is None and L.revive_server(m, "x") is False
    assert L.loaded_models_of(m) == [] and L.unload_model(m, "x") is False and L.unload_all_models(m) == []
    assert L.set_read_timeout(m, 5) is False and L.streaming_chat(m) is None
    assert L.server_url_of(None, "x") is None                                               # even no client at all


def test_llm_seam_helpers_swallow_a_broken_backend():
    from agent import llm as L

    class Broken:
        def server_url(self, model):
            raise RuntimeError("x")

        def loaded_models(self):
            raise RuntimeError("x")

        def unload(self, model):
            raise RuntimeError("x")

        def unload_all(self):
            raise RuntimeError("x")

    b = Broken()
    assert L.server_url_of(b, "m") is None and L.loaded_models_of(b) == [] and L.unload_model(b, "m") is False and L.unload_all_models(b) == []


def test_llm_seam_helpers_use_the_methods_when_present():
    from agent import llm as L

    class Full(MinimalLLM):
        def server_url(self, model):
            return "http://h:1"

        def loaded_models(self):
            return ["a", "b"]

        def unload(self, model):
            return True

        def unload_all(self):
            return ["a"]

        def set_read_timeout(self, s):
            self.t = s

        def chat_with_tools_stream(self, *a, **k):
            return {}

    f = Full()
    assert L.server_url_of(f, "m") == "http://h:1" and L.loaded_models_of(f) == ["a", "b"] and L.unload_model(f, "a") is True
    assert L.unload_all_models(f) == ["a"] and L.set_read_timeout(f, 7) is True and f.t == 7 and L.streaming_chat(f) is not None


def test_session_helpers_survive_a_backend_without_ollama_methods():
    import io
    from rich.console import Console
    from agent.codeagent import session as S

    out = io.StringIO()
    console = Console(file=out, width=100, force_terminal=False)
    m = MinimalLLM()
    S.free_gpu(console, m, "some-model")
    S.release_gpu_on_exit(console, m, {"ollama": {"unload_on_exit": True}})
    S.apply_client_config(m, {"coding": {"read_timeout_s": 5}})
    assert out.getvalue() == ""


def test_engine_turn_and_server_wait_with_a_minimal_backend(tmp_path):
    from tests.test_codeagent_engine import call, engine, say
    llm = MinimalLLM([call("write_file", path="a.py", content="x = 1\n"), say("Created a.py.")])
    eng, repo, _out = engine(tmp_path, llm)
    assert eng._await_server() is True                                                      # no health endpoint: just resend
    eng.turn("make a.py")
    assert (repo / "a.py").read_text() == "x = 1\n"


def test_supervisor_server_ok_without_server_url():
    from agent.codeagent import supervisor as SV

    class Sess:
        llm = MinimalLLM()
        model = "m"

    sup = SV.Supervisor.__new__(SV.Supervisor)
    sup.session, sup.probe = Sess(), None
    assert sup.server_ok() is True


def test_engine_core_names_no_ollama_endpoints_or_options():
    """Ollama specifics live in llm.py / serverwait.py; the engine files must not talk to `/api/` or Ollama-only settings."""
    import re
    from pathlib import Path
    import agent.codeagent as pkg

    bad = re.compile(r"/api/|keep_alive|OLLAMA_HOST")
    for f in Path(pkg.__file__).parent.glob("*.py"):
        if f.name == "serverwait.py":
            continue
        assert not bad.search(f.read_text(encoding="utf-8")), f.name
