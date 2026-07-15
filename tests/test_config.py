from pathlib import Path

import yaml

from agent.config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, load_config


def _key_paths(d: dict, prefix: str = "") -> set[str]:
    """Flatten a nested dict into a set of dotted key paths (leaves only)."""
    paths = set()
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths |= _key_paths(value, path)
        else:
            paths.add(path)
    return paths


def test_checked_in_config_has_all_default_keys() -> None:
    with DEFAULT_CONFIG_PATH.open() as f:
        checked_in = yaml.safe_load(f)

    missing = _key_paths(DEFAULT_CONFIG) - _key_paths(checked_in)
    assert not missing, f"config.yaml is missing keys present in DEFAULT_CONFIG: {missing}"


def test_creates_default_config_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    assert not config_path.exists()

    config = load_config(config_path)

    assert config_path.exists()
    assert config == DEFAULT_CONFIG


def test_loads_existing_config_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  chat: "custom-model"
  background: "llama3.1:8b"
  embed: "nomic-embed-text"
persona:
  name: "Buddy"
  style: "You are a helpful friend."
"""
    )

    config = load_config(config_path)

    assert config["models"]["chat"] == "custom-model"
    assert config["persona"]["name"] == "Buddy"


def test_creates_data_dir_alongside_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    assert not data_dir.exists()

    load_config(config_path, data_dir=data_dir)

    assert data_dir.is_dir()


def test_missing_keys_are_filled_from_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models:\n  chat: x\n")

    config = load_config(config_path)

    assert config["models"]["chat"] == "x"
    assert config["data"]["db_path"] == DEFAULT_CONFIG["data"]["db_path"]
    assert config["persona"] == DEFAULT_CONFIG["persona"]


def test_default_config_is_not_returned_by_reference(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"

    config = load_config(config_path)
    config["persona"]["name"] = "Mutated"

    assert DEFAULT_CONFIG["persona"]["name"] != "Mutated"
