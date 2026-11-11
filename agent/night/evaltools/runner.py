"""Shared helpers for the evaltools: project copy, bounded subprocess runs,
output normalisation, one-JSON-line output, common CLI arguments."""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SKIP_DIRS = {".git", ".night", ".scratch", "__pycache__", "node_modules", ".venv", "venv"}
MAX_COPY_FILE = 5 * 1024 * 1024
CLEAN_EXIT = {0, 1, 130}


def sha1(data):
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    return hashlib.sha1(data).hexdigest()


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def log(*a):
    print(*a, file=sys.stderr)


def common_args(p, need_store=True):
    p.add_argument("--root", required=True)
    p.add_argument("--store", required=True)
    return p


def setup(parser, args):
    """Validate --root/--store (usage error = exit 2). Returns (root, store)."""
    root = Path(args.root).resolve()
    store = Path(args.store).resolve()
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")
    if store == root or root in store.parents:
        parser.error("--store must be outside the project root")
    store.mkdir(parents=True, exist_ok=True)
    return root, store


def copy_project(root, dst):
    def ignore(d, names):
        out = []
        for n in names:
            p = os.path.join(d, n)
            if n in SKIP_DIRS:
                out.append(n)
            elif os.path.isfile(p) and not os.path.islink(p) and os.path.getsize(p) > MAX_COPY_FILE:
                out.append(n)
        return out
    shutil.copytree(root, dst, ignore=ignore, symlinks=True)


def run(argv, cwd, stdin="", timeout=20, env=None, shell=False):
    """Returns (returncode, merged output text, timed_out). Kills the whole process group on timeout."""
    p = subprocess.Popen(argv, cwd=cwd, env=env, shell=shell, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    timed_out = False
    try:
        out, _ = p.communicate(stdin.encode("utf-8", "replace"), timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = p.communicate()
    return p.returncode, out.decode("utf-8", "replace"), timed_out


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_]")
_TIME = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
_ADDR = re.compile(r"0x[0-9a-fA-F]{6,}")


def normalise(text, roots=()):
    for r in sorted({str(r) for r in roots if r}, key=len, reverse=True):
        text = text.replace(r, "<ROOT>")
    text = _ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _TIME.sub("HH:MM:SS", text)
    text = _ADDR.sub("0xADDR", text)
    return "\n".join(l.rstrip() for l in text.split("\n")).rstrip("\n") + "\n"


def launcher(cmd, seed):
    """Wrap `python3 [flags] main.py args` (or `-m mod args`) so random is seeded; None if not a python command."""
    t = shlex.split(cmd)
    if len(t) < 2:
        return None
    if t[1] == "-m" and len(t) > 2:
        pre, target, call = [], t[2:], "runpy.run_module(a[0], run_name='__main__', alter_sys=True)"
    else:
        i = next((k for k, x in enumerate(t) if k and x.endswith(".py")), None)
        if i is None:
            return None
        pre, target, call = t[1:i], t[i:], "runpy.run_path(a[0], run_name='__main__')"
    code = f"import random,runpy,sys; random.seed({int(seed)}); a=sys.argv[1:]; sys.argv=a; {call}"
    return [t[0], *pre, "-c", code, *target]


def run_session(root, cmd, seed, lines, timeout):
    """One deterministic session in a throwaway COPY of the project. Returns dict(rc, out, timed_out)."""
    tmp = Path(tempfile.mkdtemp(prefix="evaltools-")).resolve()
    try:
        proj = tmp / "proj"
        copy_project(root, proj)
        (tmp / "home").mkdir()
        env = dict(os.environ, PYTHONHASHSEED="0", TERM="dumb", COLUMNS="100", LINES="40",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   HOME=str(tmp / "home"), XDG_CONFIG_HOME=str(tmp / "home" / ".config"),
                   XDG_DATA_HOME=str(tmp / "home" / ".local"))
        argv = launcher(cmd, seed) or shlex.split(cmd)  # shortcut: non-python cmds run unseeded
        rc, out, to = run(argv, str(proj), "".join(l + "\n" for l in lines), timeout, env)
        return {"rc": rc, "timed_out": to, "out": normalise(out, [proj, tmp, root])}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def map_sessions(fn, items, workers=4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))
