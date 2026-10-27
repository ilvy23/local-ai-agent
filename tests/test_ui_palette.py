from agent.ui.commands import REGISTRY
from agent.ui.palette import pick_command, score_command


def names(q, limit=None):
    return [m.command.name for m in pick_command(REGISTRY, q, limit)]


def test_empty_query_lists_everything_in_registry_order():
    assert names("") == [c.name for c in REGISTRY.commands()]
    assert names("/") == names("")


def test_exact_beats_prefix_beats_substring():
    assert names("status")[0] == "status"
    assert names("sta")[0] == "status"       # prefix
    r = names("pla")
    assert r[0] == "plan"
    assert names("slow")[0] == "why-slow"     # word part of a hyphenated name


def test_alias_matches_and_exact_name_wins_over_alias():
    assert names("ctx")[0] == "context"
    assert names("exit")[0] == "quit"
    m = pick_command(REGISTRY, "quit")[0]
    assert m.score > pick_command(REGISTRY, "exit")[0].score


def test_fuzzy_subsequence_and_summary_words():
    assert "transcript" in names("trnscr")
    assert "doctor" in names("dctr")
    assert "export" in names("transcript as markdown") or "export" in names("markdown")


def test_no_match_and_limit_and_slash_prefix():
    assert names("zzzzqq") == []
    assert len(names("s", limit=3)) == 3
    assert names("/status")[0] == "status"


def test_pattern_commands_insert_a_dash_and_show_dir_placeholder():
    m = pick_command(REGISTRY, "auto")[0]
    assert m.name == "/auto-<dir>" and m.insert == "/auto-"
    assert pick_command(REGISTRY, "help")[0].insert == "/help "


def test_ties_prefer_shorter_name_then_alphabetical():
    r = pick_command(REGISTRY, "re")
    keys = [(-m.score, len(m.command.name), m.command.name) for m in r]
    assert keys == sorted(keys)


def test_score_is_zero_for_unrelated():
    assert score_command(REGISTRY.get("stop"), "xyz")[0] == 0
