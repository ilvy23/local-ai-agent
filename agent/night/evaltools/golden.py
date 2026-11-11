"""Golden-master behaviour comparison: python -m agent.night.evaltools.golden {record,compare}."""
import argparse
import json
import random
import sys

from . import runner as R

POOL = ["1", "2", "3", "4", "5", "y", "n", "", "look", "inventory", "journal", "help",
        "talk", "go", "north", "back", "quit", "0"]


def inputs(seed, i):
    r = random.Random(seed * 1000 + i)
    return [r.choice(POOL) for _ in range(r.randint(10, 80))]


def _run(root, cmd, seed, i, timeout):
    lines = inputs(seed, i)
    s = R.run_session(root, cmd, seed * 1000 + i, lines, timeout)
    s["inputs_sha"] = R.sha1("\n".join(lines))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(prog="golden")
    sp = ap.add_subparsers(dest="sub", required=True)
    rec = R.common_args(sp.add_parser("record"))
    rec.add_argument("--cmd", required=True)
    rec.add_argument("--sessions", type=int, default=12)
    rec.add_argument("--seed", type=int, default=3)
    rec.add_argument("--timeout", type=float, default=20)
    cmp_ = R.common_args(sp.add_parser("compare"))
    cmp_.add_argument("--cmd")
    cmp_.add_argument("--timeout", type=float, default=20)
    a = ap.parse_args(argv)
    root, store = R.setup(ap, a)
    gf = store / "golden.json"

    if a.sub == "record":
        if a.sessions < 1:
            ap.error("--sessions must be >= 1")
        res = R.map_sessions(lambda i: _run(root, a.cmd, a.seed, i, a.timeout), range(a.sessions))
        (store / "golden").mkdir(exist_ok=True)
        sess, crashed = [], 0
        for i, s in enumerate(res):
            (store / "golden" / f"{i}.txt").write_text(s["out"], encoding="utf-8")
            if s["timed_out"] or "Traceback (most recent call last)" in s["out"]:
                crashed += 1
            sess.append({"session": i, "seed": a.seed * 1000 + i, "inputs_sha": s["inputs_sha"],
                         "output_sha": R.sha1(s["out"]), "exit_code": s["rc"],
                         "timed_out": s["timed_out"], "lines": s["out"].count("\n"),
                         "first_lines": s["out"].split("\n")[:5]})
        gf.write_text(json.dumps({"cmd": a.cmd, "seed": a.seed, "timeout": a.timeout,
                                  "sessions": sess}, indent=1))
        R.emit({"sessions": a.sessions, "recorded": len(sess), "crashed": crashed})
        return 0

    if not gf.is_file():
        print("no golden.json in store; run record first", file=sys.stderr)
        return 2
    g = json.loads(gf.read_text())
    cmd = a.cmd or g["cmd"]
    res = R.map_sessions(lambda s: _run(root, cmd, g["seed"], s["session"], a.timeout), g["sessions"])
    different, first = [], None
    for s, new in zip(g["sessions"], res):
        i = s["session"]
        if R.sha1(new["out"]) == s["output_sha"] and new["rc"] == s["exit_code"]:
            continue
        different.append(i)
        if first is None:
            old = (store / "golden" / f"{i}.txt").read_text(encoding="utf-8").split("\n")
            nw = new["out"].split("\n")
            k = next((k for k in range(max(len(old), len(nw)))
                      if (old[k] if k < len(old) else None) != (nw[k] if k < len(nw) else None)), None)
            if k is None:
                first = {"session": i, "line": 0, "was": f"exit {s['exit_code']}", "now": f"exit {new['rc']}"}
            else:
                first = {"session": i, "line": k + 1,
                         "was": (old[k] if k < len(old) else "")[:200],
                         "now": (nw[k] if k < len(nw) else "")[:200]}
    R.emit({"sessions": len(g["sessions"]), "identical": len(g["sessions"]) - len(different),
            "different": different, "first_diff": first})
    return 0


if __name__ == "__main__":
    sys.exit(main())
