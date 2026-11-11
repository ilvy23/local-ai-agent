"""Robustness fuzzing with hostile input: python -m agent.night.evaltools.fuzz run."""
import argparse
import random
import re
import string
import sys
from collections import Counter

from . import runner as R

TB = "Traceback (most recent call last)"
_FILE = re.compile(r'File "([^"]+)", line (\d+)')
_GARBAGE = string.digits + string.ascii_letters + string.punctuation + " "
_UNI = ["héllo", "日本語", "\U0001F600" * 5, "‮evil", "́" * 8, "ßİ"]
_NUM = ["-1", "99999", "3.14", "0", "-0", "1e9", "nan", "007", "+5", "2147483648"]


def hostile_line(r):
    k = r.randrange(8)
    if k == 0:
        return "".join(r.choice(_GARBAGE) for _ in range(r.randint(1, 40)))
    if k == 1:
        return "".join(r.choice(_GARBAGE) for _ in range(2000))
    if k == 2:
        return r.choice(_UNI)
    if k == 3:
        return "".join(chr(r.choice([*range(0, 10), *range(11, 32), 127])) for _ in range(r.randint(1, 6)))
    if k == 4:
        return ""
    if k == 5:
        return r.choice(_NUM)
    if k == 6:
        return "\x1b[A\x1b[2J"
    return r.choice(["quit", "1", "2", "y", "look"])


def session_lines(seed, i):
    r = random.Random(seed * 1000 + i)
    lines = []
    for _ in range(r.randint(10, 60)):
        lines += [""] * r.randint(3, 8) if r.random() < 0.05 else [hostile_line(r)]
    if r.random() < 0.5:  # Ctrl+D: EOF at a random point
        lines = lines[:r.randrange(len(lines))]
    return lines


def signature(out, rc, timed_out):
    if timed_out:
        return "timeout (hang) @ -"
    n = out.rfind(TB)
    if n < 0:
        return f"exit code {rc} @ -"
    where, exc, seen = "?", "?", False
    for ln in out[n:].split("\n")[1:]:
        m = _FILE.search(ln)
        if m:
            path = m.group(1).replace("<ROOT>/", "")
            where, seen = f"{path.rsplit('/', 1)[-1] if path.startswith('/') else path}:{m.group(2)}", True
        elif seen and ln.strip() and not ln[0].isspace():
            exc = ln.strip()
            break
    return f"{exc[:80]} @ {where}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fuzz")
    sp = ap.add_subparsers(dest="sub", required=True)
    p = R.common_args(sp.add_parser("run"))
    p.add_argument("--cmd", required=True)
    p.add_argument("--sessions", type=int, default=40)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--timeout", type=float, default=20)
    a = ap.parse_args(argv)
    root, store = R.setup(ap, a)
    if a.sessions < 1:
        ap.error("--sessions must be >= 1")

    def one(i):
        return R.run_session(root, a.cmd, a.seed * 1000 + i, session_lines(a.seed, i), a.timeout)

    sigs, crashes, hangs = Counter(), 0, 0
    for s in R.map_sessions(one, range(a.sessions)):
        bad = s["timed_out"] or TB in s["out"] or s["rc"] not in R.CLEAN_EXIT
        hangs += s["timed_out"]
        if bad:
            crashes += 1
            sigs[signature(s["out"], s["rc"], s["timed_out"])] += 1
    R.emit({"sessions": a.sessions, "crashes": crashes, "hangs": hangs,
            "signatures": dict(sigs.most_common(10))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
