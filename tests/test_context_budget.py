from agent.context.budget import ContextBudget, estimate_tokens


def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_nonempty_is_at_least_one():
    assert estimate_tokens("x") == 1


def test_estimate_scales_with_length():
    assert estimate_tokens("a" * 400) == 100


def test_add_accumulates_within_section():
    b = ContextBudget(limit=1000)
    b.add("files", "a" * 400)
    b.add("files", "a" * 400)
    assert b.sections["files"] == 200


def test_add_accepts_precounted_int():
    b = ContextBudget(limit=1000)
    assert b.add("system", 42) == 42


def test_total_and_fraction():
    b = ContextBudget(limit=100)
    b.add("system", 25)
    b.add("history", 25)
    assert b.total == 50
    assert b.fraction == 0.5


def test_over_threshold():
    b = ContextBudget(limit=100)
    b.add("files", 80)
    assert b.over(0.75)
    assert not b.over(0.9)


def test_reset_drops_section():
    b = ContextBudget(limit=100)
    b.add("repomap", 30)
    b.reset("repomap")
    assert "repomap" not in b.sections
    assert b.total == 0


def test_render_orders_sections_and_shows_total():
    b = ContextBudget(limit=16384)
    b.add("history", 2100)
    b.add("system", 412)
    b.add("files", 4890)
    out = b.render()
    lines = out.splitlines()
    assert lines[0].startswith("system")
    assert "files" in out
    assert "total" in lines[-1]
    assert "16,384" in lines[-1]


def test_fraction_zero_when_limit_zero():
    assert ContextBudget(limit=0).fraction == 0.0
