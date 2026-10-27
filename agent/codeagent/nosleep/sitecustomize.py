"""Put on PYTHONPATH by `run(..., no_sleep=true)` only: time.sleep returns at once, so a text game that types its output
slowly, or a script that pauses between steps, runs in a second instead of 20 - same code path, no waiting.

This file shadows the interpreter's own sitecustomize (NixOS and some distros ship one that sets up their paths), so that
one is looked up further down sys.path and run as well.
"""

import os
import sys
import time


def _no_sleep(*_args, **_kwargs):
    return None


time.sleep = _no_sleep


def _chain() -> None:
    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    path = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is None or spec.loader is None or not spec.origin or os.path.dirname(os.path.abspath(spec.origin)) == here:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


try:
    _chain()
except Exception:  # noqa: BLE001 - a broken system sitecustomize must not stop the program (the normal startup ignores it too)
    pass
