from agent.tools.system import system_stats


def test_system_stats_returns_readable_string():
    out = system_stats()
    assert isinstance(out, str)
    lower = out.lower()
    assert "ram" in lower or "memory" in lower
    assert "disk" in lower
    # Should mention some figures (GB/MB) and never raise.
    assert "b" in lower  # GB/MB/B units present


def test_system_stats_mentions_processes():
    out = system_stats().lower()
    assert "process" in out or "top" in out
