import pytest

from agent import repoaccess
from agent.ui.commands import GROUPS, REGISTRY, Command, Registry


def test_seeded_names():
    want = ("help quit exit status context tasks queue log events usage transcript thinking stop pause resume compact "
            "steps mode say model clear new undo notes plan project verify access export doctor why-slow web").split()
    for n in want:
        assert REGISTRY.get(n) is not None, n
    assert {g for g, _ in REGISTRY.help_groups()} <= set(GROUPS)
    assert all(c.summary and c.usage for c in REGISTRY.commands())


EXPECT = {
    "instant": ["/status", "/context", "/tasks", "/queue", "/queue rm 2", "/log", "/usage", "/help", "/transcript",
                "/thinking off", "/stop", "/pause", "/resume", "/export", "/events", "/why-slow", "/undo list", "/plan"],
    "step": ["/compact", "/steps 30", "/mode auto", "/say use pytest"],
    "deferred": ["/clear", "/new", "/model", "/undo", "/quit", "/exit", "/web cats", "/plan clear", "/auto-~/x"],
}


@pytest.mark.parametrize("when", EXPECT)
def test_classify_busy_and_idle(when):
    for text in EXPECT[when]:
        idle = REGISTRY.classify(text, busy=False)
        assert idle.action == "run_now", text
        d = REGISTRY.classify(text, busy=True)
        want = {"instant": "run_now", "step": "step", "deferred": "defer"}[when]
        assert d.action == want, (text, d)
        if want != "run_now":
            assert d.notice


def test_not_a_command_and_unknown():
    assert REGISTRY.classify("hello", True).action == "not_a_command"
    assert REGISTRY.classify("/home/me/proj fix it", True).action == "not_a_command"
    d = REGISTRY.classify("/bogus", False)
    assert d.action == "refuse" and "bogus" in d.notice
    assert REGISTRY.classify("/", False).action == "not_a_command"


def test_aliases_and_args():
    cmd, args = REGISTRY.resolve("/ctx")
    assert cmd.name == "context" and args == ""
    cmd, args = REGISTRY.resolve("/s  rust lifetimes ")
    assert cmd.name == "web" and args == "rust lifetimes"
    assert REGISTRY.resolve("/QUEUE rm 1")[0].name == "queue"
    assert REGISTRY.resolve("nope") is None


@pytest.mark.parametrize("text,name,arg", [
    ("/auto-~/code/x", "auto", "~/code/x"), ("/medium-/tmp/p", "medium", "/tmp/p"),
    ("/greenlist-/srv/a b", "greenlist", "/srv/a b"), ("/ungreenlist-/srv/a", "ungreenlist", "/srv/a"),
    ("/auto /tmp/p", "auto", "/tmp/p"), ("/auto", "auto", ""), ("/ask-/tmp/p", "ask", "/tmp/p"),
])
def test_pattern_commands(text, name, arg):
    cmd, args = REGISTRY.resolve(text)
    assert (cmd.name, args) == (name, arg) and cmd.pattern
    # stays in step with the real regex
    assert repoaccess.COMMAND_RE.match(text)


def test_autopilot_is_not_auto_pattern():
    assert REGISTRY.resolve("/autopilot") is None


def test_completions():
    names = [c.name for c in REGISTRY.completions("/st")]
    assert names[:3] == ["/status", "/steps", "/stop"] or {"/status", "/steps", "/stop"} <= set(names[:4])
    assert REGISTRY.completions("/")  # everything
    assert REGISTRY.completions("slow")[0].name == "/why-slow"       # substring
    assert "/transcript" in [c.name for c in REGISTRY.completions("tscr")]  # subsequence
    assert REGISTRY.completions("zzzz") == []
    assert "/auto-<dir>" in [c.name for c in REGISTRY.completions("aut")]
    assert REGISTRY.completions("ctx")[0].name == "/context"


def test_custom_registry_and_bind():
    r = Registry()
    r.register(Command("x", "s", "/x", "Chat", "instant"))
    called = []
    r.bind("x", lambda a: called.append(a))
    r.resolve("/x 1")[0].handler("1")
    assert called == ["1"]
    with pytest.raises(ValueError):
        r.register(Command("y", when="whenever"))
