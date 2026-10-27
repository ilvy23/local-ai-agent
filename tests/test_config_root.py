"""Where config.yaml and data/ live: next to the code in a checkout, in a per-user folder after a plain `pip install`."""
import importlib
from pathlib import Path

import agent.config as C


def _reload_with(monkeypatch, home=None):
    if home is None:
        monkeypatch.delenv("NIGHTAGE_HOME", raising=False)
    else:
        monkeypatch.setenv("NIGHTAGE_HOME", str(home))
    return importlib.reload(C)


def test_checkout_keeps_config_and_data_next_to_the_code(monkeypatch):
    try:
        mod = _reload_with(monkeypatch)
        assert (mod.PROJECT_ROOT / "pyproject.toml").exists()
        assert mod.DEFAULT_CONFIG_PATH == mod.PROJECT_ROOT / "config.yaml" and mod.DEFAULT_DATA_DIR == mod.PROJECT_ROOT / "data"
    finally:
        monkeypatch.undo()
        importlib.reload(C)


def test_nightage_home_moves_them_to_a_user_folder(monkeypatch, tmp_path):
    try:
        mod = _reload_with(monkeypatch, tmp_path / "home")
        assert mod.PROJECT_ROOT == tmp_path / "home" and mod.DEFAULT_DATA_DIR == tmp_path / "home" / "data"
        cfg = mod.load_config()
        assert (tmp_path / "home" / "config.yaml").exists() and cfg["models"]["chat"]
    finally:
        monkeypatch.undo()
        importlib.reload(C)
    assert isinstance(C.PROJECT_ROOT, Path)
