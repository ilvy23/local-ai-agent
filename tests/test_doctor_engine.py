"""The doctor engine: registry, isolation, timeouts, dependency skips, system filtering, plugin checks. Fakes only."""

import threading
import time

import pytest

from agent import sysinfo
from agent.doctor import core
from agent.doctor.core import Context, check, fail, ok, skip, warn
from tests._doctorfake import CFG, UP, FakeProbes, ctx_for


@pytest.fixture
def isolated_registry(monkeypatch):
    """A private registry so test checks never leak into the real one."""
    monkeypatch.setattr(core, "REGISTRY", {})
    monkeypatch.setattr(core, "_discovered", True)
    return core.REGISTRY


def run(ctx, **kw):
    return core.run_checks(ctx, core.select(ctx, **kw))


def test_decorator_registers_and_runs(tmp_path, isolated_registry):
    @check("a.one", "python", "One")
    def one(ctx):
        return ok("fine")

    res = run(ctx_for(tmp_path))
    assert [(r.id, r.group, r.title, r.status, r.detail) for r in res] == [("a.one", "python", "One", "ok", "fine")]


def test_exception_becomes_could_not_run_not_a_crash(tmp_path, isolated_registry):
    @check("boom", "python", "Boom")
    def boom(ctx):
        raise RuntimeError("kaput")

    @check("after", "python", "After")
    def after(ctx):
        return ok("still ran")

    res = run(ctx_for(tmp_path))
    assert res[0].status == "warn" and "could not run" in res[0].detail and "kaput" in res[0].detail
    assert res[1].status == "ok"


def test_systemexit_in_a_check_is_contained(tmp_path, isolated_registry):
    @check("exit", "python", "Exit")
    def ex(ctx):
        raise SystemExit(3)

    assert run(ctx_for(tmp_path))[0].status == "warn"


def test_timeout_skips_the_slow_check_and_does_not_block(tmp_path, isolated_registry):
    release = threading.Event()

    @check("slow", "python", "Slow", timeout=0.2)
    def slow(ctx):
        release.wait(5)
        return ok("late")

    t0 = time.monotonic()
    res = run(ctx_for(tmp_path))
    release.set()
    assert time.monotonic() - t0 < 2
    assert res[0].status == "warn" and "within 0.2 s" in res[0].detail


def test_network_checks_run_in_parallel(tmp_path, isolated_registry):
    barrier = threading.Barrier(3, timeout=3)

    for i in range(3):
        @check(f"net.{i}", "network", f"Net {i}", network=True, timeout=5)
        def net(ctx, i=i):
            barrier.wait()                       # only passes when all three run at the same time
            return ok(f"n{i}")

    res = run(ctx_for(tmp_path))
    assert [r.status for r in res] == ["ok"] * 3


def test_no_web_skips_network_checks_with_a_reason(tmp_path, isolated_registry):
    called = []

    @check("net", "network", "Net", network=True)
    def net(ctx):
        called.append(1)
        return ok("x")

    res = run(ctx_for(tmp_path, no_web=True))
    assert res[0].status == "skip" and "--no-web" in res[0].detail and not called


def test_needs_ollama_skips_with_reason(tmp_path, isolated_registry):
    @check("m", "models", "M", needs_ollama=True)
    def m(ctx):
        return ok("never")

    down = run(ctx_for(tmp_path))
    assert down[0].status == "skip" and "Ollama" in down[0].detail
    up = run(ctx_for(tmp_path, ollama=UP))
    assert up[0].status == "ok"


def test_requires_skips_when_the_dependency_failed(tmp_path, isolated_registry):
    @check("base", "python", "Base")
    def base(ctx):
        return fail("bad")

    @check("child", "python", "Child", requires=("base",))
    def child(ctx):
        return ok("never")

    @check("fine", "python", "Fine")
    def fine(ctx):
        return ok("y")

    @check("child2", "python", "Child2", requires=("fine",))
    def child2(ctx):
        return ok("ran")

    res = {r.id: r for r in run(ctx_for(tmp_path))}
    assert res["child"].status == "skip" and "base" in res["child"].detail
    assert res["child2"].status == "ok"


def test_requires_orders_dependencies_first(tmp_path, isolated_registry):
    order = []

    @check("late", "python", "L", requires=("early",))
    def late(ctx):
        order.append("late")
        return ok("")

    @check("early", "python", "E")
    def early(ctx):
        order.append("early")
        return ok("")

    run(ctx_for(tmp_path))
    assert order == ["early", "late"]


def test_system_filtering_by_fake_sysinfo(tmp_path, isolated_registry):
    for tag in ("linux", "macos", "windows", "termux", "nixos", "wsl", "systemd"):
        @check(f"only.{tag}", "system", tag, systems=(tag,))
        def f(ctx, tag=tag):
            return ok(tag)

    @check("everywhere", "system", "all")
    def e(ctx):
        return ok("all")

    def ids(si, **kw):
        return {r.id for r in run(ctx_for(tmp_path, si=si, **kw))}

    nixos = sysinfo.SysInfo(os_family="linux", distro="nixos", init_system="systemd")
    assert ids(nixos) == {"everywhere", "only.linux", "only.nixos", "only.systemd"}
    mac = sysinfo.SysInfo(os_family="macos")
    assert ids(mac) == {"everywhere", "only.macos"}
    phone = sysinfo.SysInfo(os_family="android-termux", is_termux=True)
    assert ids(phone) == {"everywhere", "only.termux"}
    win = sysinfo.SysInfo(os_family="windows")
    assert ids(win) == {"everywhere", "only.windows"}
    assert ids(sysinfo.SysInfo(os_family="linux", distro="ubuntu"), wsl=True) == {"everywhere", "only.linux", "only.wsl"}


def test_deep_checks_only_with_deep(tmp_path, isolated_registry):
    @check("d", "deep", "D", deep=True)
    def d(ctx):
        return ok("deep")

    assert run(ctx_for(tmp_path)) == []
    res = run(ctx_for(tmp_path, deep=True))
    assert res[0].id == "d" and res[0].deep


def test_only_and_skip_filters_match_group_id_and_prefix(tmp_path, isolated_registry):
    for cid, grp in (("gpu", "gpu"), ("gpu.nvidia", "gpu"), ("disk.data", "disk"), ("ollama", "ollama")):
        @check(cid, grp, cid)
        def f(ctx, cid=cid):
            return ok(cid)

    ctx = ctx_for(tmp_path)
    assert {r.id for r in run(ctx, only=["gpu"])} == {"gpu", "gpu.nvidia"}
    assert {r.id for r in run(ctx, only=["disk.data"])} == {"disk.data"}
    assert {r.id for r in run(ctx, skip_groups=["gpu", "disk"])} == {"ollama"}
    assert {r.id for r in run(ctx, only=["gpu,ollama".split(",")[0]], skip_groups=["gpu.nvidia"])} == {"gpu"}


def test_multi_result_checks_and_none(tmp_path, isolated_registry):
    @check("multi", "models", "Multi")
    def multi(ctx):
        return [core.Result("ok", "a", id="multi.a"), core.Result("warn", "b", id="multi.b", title="Bee")]

    @check("nothing", "models", "Nothing")
    def nothing(ctx):
        return None

    res = run(ctx_for(tmp_path))
    assert [(r.id, r.title) for r in res] == [("multi.a", "Multi"), ("multi.b", "Bee")]


def test_results_are_sorted_by_group_order(tmp_path, isolated_registry):
    for cid, grp in (("z", "deep"), ("y", "gpu"), ("x", "python"), ("w", "customgroup")):
        @check(cid, grp, cid, deep=(grp == "deep"))
        def f(ctx, cid=cid):
            return ok(cid)

    res = run(ctx_for(tmp_path, deep=True))
    assert [r.group for r in res] == ["python", "gpu", "deep", "customgroup"]


def test_a_failing_probe_is_cached_and_isolated(tmp_path, isolated_registry):
    class Bad(FakeProbes):
        def sysinfo(self):
            raise OSError("no /proc")

    @check("needs_si", "system", "NeedsSI")
    def n(ctx):
        return ok(str(ctx.sysinfo.cpu_count))

    ctx = Context(CFG, Bad(tmp_path))
    assert ctx.tags == frozenset()                    # unknown system: only the checks that apply everywhere are selected
    res = run(ctx)
    assert res[0].status == "warn" and "could not run" in res[0].detail and "no /proc" in res[0].detail


def test_context_shared_facts_are_computed_once(tmp_path):
    calls = []

    class Counting(FakeProbes):
        def ollama(self):
            calls.append(1)
            return UP

    ctx = Context(CFG, Counting(tmp_path))
    threads = [threading.Thread(target=lambda: ctx.ollama) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1


def test_real_registry_discovers_every_area_and_has_unique_ids():
    core.discover()
    groups = {s.group for s in core.REGISTRY.values()}
    assert {"python", "packages", "ollama", "models", "gpu", "memory", "disk", "network", "ports", "terminal", "filesystem",
            "database", "config", "security", "plugins", "system", "deep"} <= groups
    assert core.LOAD_ERRORS == []
    assert all(s.title for s in core.REGISTRY.values())
    assert {s.id for s in core.REGISTRY.values() if s.deep} == {"deep.generate", "deep.embed", "deep.toolcall", "deep.recommend"}
    systems = {t for s in core.REGISTRY.values() for t in (s.systems or ())}
    assert {"linux", "nixos", "termux", "macos", "windows", "wsl"} <= systems


def test_plugin_result_coercion():
    assert core.coerce_plugin_result("hello").status == "ok"
    r = core.coerce_plugin_result(("warn", "slow", "fix it"))
    assert (r.status, r.detail, r.hint) == ("warn", "slow", "fix it")
    assert core.coerce_plugin_result({"status": "fail", "detail": "x"}).status == "fail"
    assert core.coerce_plugin_result(("nonsense", "x")).status == "warn"
    assert core.coerce_plugin_result(42).status == "warn"


def test_skip_helper_and_dict_shape():
    r = skip("because")
    r.id, r.group, r.title = "i", "g", "t"
    d = r.to_dict()
    assert d["status"] == "skip" and set(d) >= {"id", "group", "title", "status", "detail", "hint", "command", "ms"}
    assert "hint" not in r.to_dict(hints=False)
    assert warn("x").status == "warn"
