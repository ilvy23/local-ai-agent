import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEST_CMD = shlex.join([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
GAME_CMD = f"{shlex.quote(sys.executable)} main.py --no-music"

RULES = '''def classify(n):
    if n == 0:
        return "zero"
    if n > 3:
        return "big"
    return "small"


def add(a, b):
    return a + b


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def both(a, b):
    if a and b:
        return "both"
    return "no"
'''

MAIN = '''import random
import sys
import time
from rules import add, classify


def main():
    total = 0
    try:
        while True:
            line = input("> ").strip()
            if line == "quit":
                print("Goodbye")
                break
            try:
                n = int(line)
            except ValueError:
                n = None
            if n is not None:
                total = add(total, n)
                print("class", classify(n), "total", total)
            elif line == "look":
                print("room", random.randint(1, 1000), time.strftime("%H:%M:%S"))
            else:
                print("unknown:", line)
    except EOFError:
        print("bye")
    with open("save.json", "w") as f:
        f.write("{}")


if __name__ == "__main__":
    main()
'''

TESTS = '''from rules import add, both, clamp, classify


def test_classify():
    assert classify(0) == "zero"
    assert classify(1) == "small"
    assert classify(3) == "small"
    assert classify(4) == "big"


def test_add():
    assert add(2, 3) == 5
    assert add(0, -1) == -1


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert clamp(2, 0, 3) == 2
    assert clamp(0, 0, 3) == 0
    assert clamp(3, 0, 3) == 3


def test_both():
    assert both(True, True) == "both"
    assert both(True, False) == "no"
    assert both(False, True) == "no"
'''


def make_game(base, crash=False):
    p = base / "proj"
    (p / "tests").mkdir(parents=True)
    (p / "rules.py").write_text(RULES)
    main = MAIN
    if crash:
        main = main.replace('            if line == "quit":', '            if line == "99999":\n                raise ValueError("boom")\n            if line == "quit":')
    (p / "main.py").write_text(main)
    (p / "tests" / "test_rules.py").write_text(TESTS)
    (p / "save.json").write_text('{"real": true}')
    return p


def call(mod, *args):
    r = subprocess.run([sys.executable, "-m", f"agent.night.evaltools.{mod}", *map(str, args)],
                       cwd=REPO, capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, PYTHONPATH=str(REPO)))
    if r.returncode != 0:
        return r.returncode, None
    lines = r.stdout.strip().split("\n")
    assert len(lines) == 1, r.stdout  # exactly one stdout line
    return 0, json.loads(lines[0])


def ok(mod, *args):
    rc, j = call(mod, *args)
    assert rc == 0
    return j


def tree(p):
    return {str(f.relative_to(p)): f.read_bytes() for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts}


def test_mutate_inject_score_restore(tmp_path):
    proj, store = make_game(tmp_path), tmp_path / "store"
    before = tree(proj)
    a = ["--root", proj, "--store", store]
    j = ok("mutate", "inject", *a, "--n", 3, "--seed", 7, "--test", TEST_CMD)
    assert j["injected"] == 3 and j["tests_red"] and j["tries"] <= 40
    st = json.loads((store / "mutations.json").read_text())
    ms = st["mutations"]
    assert [m["id"] for m in ms] == j["ids"]
    assert len({(m["file"], m["function"]) for m in ms}) == 3  # never two in one function
    assert all(m["file"] == "rules.py" and m["original_line"] != m["mutated_line"] for m in ms)
    assert all(m["mutated_line"] in (proj / m["file"]).read_text() for m in ms)
    assert all(k.startswith("tests/") for k in st["tests_sha1"]) and st["tests_sha1"]
    after = tree(proj)
    assert {k for k in after if after[k] != before[k]} == {"rules.py"}  # only source mutated
    assert (store / "originals" / next(iter(st["originals"].values()))).is_file()

    s = ok("mutate", "score", *a, "--test", TEST_CMD)
    assert s["injected"] == 3 and s["fixed"] == 0 and not s["tests_green"] and not s["tests_changed"]
    assert s["unfixed"] == j["ids"]

    m = ms[0]  # manually revert one line
    p = proj / m["file"]
    ls = p.read_text().split("\n")
    ls[m["lineno"] - 1] = m["original_line"]
    p.write_text("\n".join(ls))
    s = ok("mutate", "score", *a, "--test", TEST_CMD)
    assert s["fixed"] == 1 and m["id"] not in s["unfixed"]

    t = proj / "tests" / "test_rules.py"  # cheating: weaken a test
    t.write_text(t.read_text() + "\n# weakened\n")
    assert ok("mutate", "score", *a, "--test", TEST_CMD)["tests_changed"] is True
    t.write_text(TESTS)

    assert ok("mutate", "restore", *a)["restored"] >= 1
    assert tree(proj) == before
    s = ok("mutate", "score", *a, "--test", TEST_CMD)
    assert s["fixed"] == 3 and s["tests_green"] and not s["tests_changed"]
    assert str(store) not in str(list(proj.rglob("*")))


def test_golden_record_compare_and_determinism(tmp_path):
    proj = make_game(tmp_path)
    s1, s2 = tmp_path / "s1", tmp_path / "s2"
    args = ["--cmd", GAME_CMD, "--timeout", 20]
    r = ok("golden", "record", "--root", proj, "--store", s1, *args, "--sessions", 6, "--seed", 3)
    assert r == {"sessions": 6, "recorded": 6, "crashed": 0}
    ok("golden", "record", "--root", proj, "--store", s2, *args, "--sessions", 6, "--seed", 3)
    g1, g2 = (json.loads((s / "golden.json").read_text())["sessions"] for s in (s1, s2))
    assert [x["output_sha"] for x in g1] == [x["output_sha"] for x in g2]
    assert all(x["inputs_sha"] and x["first_lines"] for x in g1)
    assert (s1 / "golden" / "0.txt").is_file()
    assert (proj / "save.json").read_text() == '{"real": true}'  # copy was used

    c = ok("golden", "compare", "--root", proj, "--store", s1, "--timeout", 20)
    assert c == {"sessions": 6, "identical": 6, "different": [], "first_diff": None}

    expected = [i for i in range(6) if "Goodbye" in (s1 / "golden" / f"{i}.txt").read_text()]
    assert expected
    (proj / "main.py").write_text(MAIN.replace('"Goodbye"', '"Farewell"'))
    c = ok("golden", "compare", "--root", proj, "--store", s1)
    assert c["different"] == expected and c["identical"] == 6 - len(expected)
    fd = c["first_diff"]
    assert fd["session"] == expected[0] and fd["was"].endswith("Goodbye") and fd["now"].endswith("Farewell") and fd["line"] > 0
    assert (proj / "save.json").read_text() == '{"real": true}'


def test_fuzz_finds_planted_crash_and_clean_game(tmp_path):
    bad = make_game(tmp_path / "a", crash=True)
    j = ok("fuzz", "run", "--root", bad, "--store", tmp_path / "sa", "--cmd", GAME_CMD,
           "--sessions", 40, "--seed", 5, "--timeout", 20)
    assert j["crashes"] > 0 and j["hangs"] == 0 and j["sessions"] == 40
    assert any(k.startswith("ValueError: boom @ main.py:") for k in j["signatures"])
    good = make_game(tmp_path / "b")
    j = ok("fuzz", "run", "--root", good, "--store", tmp_path / "sb", "--cmd", GAME_CMD,
           "--sessions", 20, "--seed", 5, "--timeout", 20)
    assert j["crashes"] == 0 and j["signatures"] == {}


def test_fuzz_counts_hang(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    (p / "main.py").write_text("while True:\n    pass\n")
    j = ok("fuzz", "run", "--root", p, "--store", tmp_path / "s", "--cmd", GAME_CMD,
           "--sessions", 2, "--timeout", 1)
    assert j["hangs"] == 2 and j["crashes"] == 2


@pytest.mark.parametrize("mod,args", [
    ("mutate", []), ("golden", []), ("fuzz", []),
    ("mutate", ["inject", "--root", "x"]),
    ("golden", ["record", "--bogus"]),
])
def test_usage_errors_exit_2(mod, args):
    assert call(mod, *args)[0] == 2


def test_store_inside_project_rejected_and_missing_state(tmp_path):
    proj = make_game(tmp_path)
    assert call("fuzz", "run", "--root", proj, "--store", proj / ".night", "--cmd", GAME_CMD)[0] == 2
    assert call("golden", "compare", "--root", proj, "--store", tmp_path / "empty")[0] == 2
    assert call("mutate", "score", "--root", proj, "--store", tmp_path / "empty", "--test", "true")[0] == 2
    assert not (proj / ".night").exists()
