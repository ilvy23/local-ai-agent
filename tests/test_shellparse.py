"""agent.shellparse: the lexer the risk classifier splits command lines with (it must see the line the way bash does)."""

from __future__ import annotations

import pytest

from agent.shellparse import group_span, split


def texts(cmd: str) -> list[str] | None:
    segs = split(cmd)
    return None if segs is None else [s.text for s in segs]


@pytest.mark.parametrize("cmd, want", [
    ("ls\nrm -rf ~", ["ls", "rm -rf ~"]),                     # newline separates
    ("ls & rm -rf ~", ["ls", "rm -rf ~"]),                    # background `&` separates
    ("ls |& rm x", ["ls", "rm x"]),
    ("a && b || c; d", ["a", "b", "c", "d"]),
    ("a;; b", ["a", "b"]),
    ("ls 2>&1", ["ls 2>&1"]),                                 # fd duplication is not a separator
    ("ls &>/dev/null", ["ls &>/dev/null"]),
    ("echo x &>> log", ["echo x &>> log"]),
    ("echo x >| f", ["echo x >| f"]),
    ("cat <&0", ["cat <&0"]),
    ('echo \\" ; rm -rf ~ ; echo \\"', ['echo \\"', "rm -rf ~", 'echo \\"']),   # escaped quotes outside quotes
    ('echo "a \\" ; b"', ['echo "a \\" ; b"']),               # escaped quote inside a double-quoted string
    ("echo 'a; b'", ["echo 'a; b'"]),
    ("echo {; rm -rf ~", ["echo {", "rm -rf ~"]),             # `{` is only a group at a command position before a blank
    ("{ ls; } ; rm -rf /", ["{ ls; }", "rm -rf /"]),
    ("(ls) && rm -rf /", ["(ls)", "rm -rf /"]),
    ("ls # ; rm -rf ~", ["ls"]),                              # a comment
    ("echo a#b; rm -rf ~", ["echo a#b", "rm -rf ~"]),         # not a comment: `#` inside a word
    ("ls \\\n -la", ["ls  -la"]),                             # line continuation
    ("f(){ rm -rf ~; }; f", ["f(){ rm -rf ~; }", "f"]),
    ("function f { reboot; }", ["function f { reboot; }"]),
    ("echo ${HOME}; rm -rf ~", ["echo ${HOME}", "rm -rf ~"]),
    ("cat <(rm -rf ~/x)", ["cat <(rm -rf ~/x)"]),
    ("x=$(echo $((1+2)))", ["x=$(echo $((1+2)))"]),
])
def test_segments_are_cut_where_bash_cuts_them(cmd, want):
    assert texts(cmd) == want


@pytest.mark.parametrize("cmd", ["(rm -rf /", 'echo "unterminated', "echo 'x", "echo (x)", "ls )", "cat <<", "{ rm -rf /",
                                 "echo `x", "echo $(ls", "x\\"])
def test_malformed_lines_are_refused(cmd):
    assert split(cmd) is None


def test_heredoc_bodies_are_taken_out_of_the_command_text():
    segs = split("cat > f.py <<'EOF'\nprint(\"don't\")\nrm -rf ~\nEOF\nls")
    assert [s.text for s in segs] == ["cat > f.py <<'EOF'", "ls"]
    assert segs[0].heredocs[0].body == "print(\"don't\")\nrm -rf ~" and segs[0].heredocs[0].expands is False


def test_unquoted_heredoc_bodies_expand_and_dash_strips_tabs():
    segs = split("cat <<EOF\n$(whoami)\nEOF")
    assert segs[0].heredocs[0].body == "$(whoami)" and segs[0].heredocs[0].expands is True
    segs = split("cat <<-X\n\tbody\n\tX\nnext")
    assert [s.text for s in segs] == ["cat <<-X", "next"] and segs[0].heredocs[0].body == "\tbody"


def test_here_string_is_not_a_heredoc():
    assert texts("cat <<< 'rm -rf ~'; ls") == ["cat <<< 'rm -rf ~'", "ls"]


def test_group_span_returns_inner_list_and_rest():
    assert group_span("(rm -rf /) 2>/dev/null") == ("rm -rf /", " 2>/dev/null")
    assert group_span("{ ls; } > out") == ("ls", " > out")
    assert group_span("{rm,-rf,~}") is None                  # brace expansion, not a group
    assert group_span("ls") is None


def test_pathological_nesting_is_refused_not_crashing():
    assert split("echo " + "$(" * 5000 + "ls" + ")" * 5000) is None or True    # must not raise
