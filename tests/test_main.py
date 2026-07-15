from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.main import app
from agent.memory.store import Store

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    """Point every test at a throwaway config + db so nothing touches the repo's data/."""
    config_path = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    monkeypatch.setattr("agent.config.DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr("agent.config.DEFAULT_DATA_DIR", data_dir)
    monkeypatch.setattr("agent.main.PROJECT_ROOT", tmp_path)
    return tmp_path


def test_sessions_lists_no_sessions_message_when_empty():
    result = runner.invoke(app, ["sessions"])

    assert result.exit_code == 0
    assert "No sessions yet" in result.output


def test_sessions_lists_existing_sessions(_isolated_db):
    db_path = _isolated_db / "data" / "agent.db"
    store = Store(db_path)
    store.create_session(title="Hello there")
    store.add_message(store.get_last_session_id(), "user", "hi")
    store.close()

    result = runner.invoke(app, ["sessions"])

    assert result.exit_code == 0
    assert "Hello there" in result.output


def test_resume_with_no_sessions_shows_friendly_error(_isolated_db):
    result = runner.invoke(app, ["resume"])

    assert result.exit_code != 0
    assert "no sessions" in result.output.lower()


def test_resume_with_unknown_session_id_shows_friendly_error(_isolated_db):
    db_path = _isolated_db / "data" / "agent.db"
    Store(db_path).close()

    result = runner.invoke(app, ["resume", "999"])

    assert result.exit_code != 0
    assert "999" in result.output


def test_memory_list_empty(_isolated_db):
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0
    assert "no facts" in result.output.lower()


def test_memory_add_then_list(_isolated_db):
    add = runner.invoke(app, ["memory", "add", "dog is named Rex"])
    assert add.exit_code == 0

    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0
    assert "Rex" in result.output


def test_memory_forget_round_trip(_isolated_db):
    db_path = _isolated_db / "data" / "agent.db"
    store = Store(db_path)
    fact_id = store.add_fact("temporary fact", source_session_id=None)
    store.close()

    forget = runner.invoke(app, ["memory", "forget", str(fact_id)])
    assert forget.exit_code == 0

    result = runner.invoke(app, ["memory", "list"])
    assert "temporary fact" not in result.output


def test_audit_empty_state(_isolated_db):
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "no" in result.output.lower() and "audit" in result.output.lower()


def test_audit_renders_rows(_isolated_db):
    db_path = _isolated_db / "data" / "agent.db"
    store = Store(db_path)
    store.add_audit_log(kind="shell", detail="ls -la", approved=1, result="exit 0")
    store.add_audit_log(kind="tool", detail="read_file", approved=1, result="read 12 chars")
    store.close()

    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "ls -la" in result.output
    assert "read_file" in result.output


def test_audit_kind_filter(_isolated_db):
    db_path = _isolated_db / "data" / "agent.db"
    store = Store(db_path)
    store.add_audit_log(kind="shell", detail="pwd", approved=1, result="ok")
    store.add_audit_log(kind="tool", detail="list_dir", approved=1, result="ok")
    store.close()

    result = runner.invoke(app, ["audit", "--kind", "shell"])
    assert result.exit_code == 0
    assert "pwd" in result.output
    assert "list_dir" not in result.output


def test_audit_limit_respected(_isolated_db):
    db_path = _isolated_db / "data" / "agent.db"
    store = Store(db_path)
    for i in range(5):
        store.add_audit_log(kind="shell", detail=f"cmd{i}", approved=1, result="ok")
    store.close()

    result = runner.invoke(app, ["audit", "--limit", "2"])
    assert result.exit_code == 0
    # Newest two only.
    assert "cmd4" in result.output
    assert "cmd3" in result.output
    assert "cmd0" not in result.output


def test_memory_search_like_fallback(_isolated_db):
    db_path = _isolated_db / "data" / "agent.db"
    store = Store(db_path)
    store.add_fact("enjoys hiking in the mountains", source_session_id=None)
    store.add_fact("drinks black coffee", source_session_id=None)
    store.close()

    # No Ollama in tests, so semantic search fails and falls back to LIKE.
    result = runner.invoke(app, ["memory", "search", "hiking"])
    assert result.exit_code == 0
    assert "hiking" in result.output
    assert "coffee" not in result.output
