"""Token protection of the web API (needs the `web` dependency group; skipped without fastapi)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.web import auth  # noqa: E402
from agent.web.app import app  # noqa: E402


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "TOKEN_PATH", tmp_path / "web-token")
    monkeypatch.delenv("AGENT_WEB_TOKEN", raising=False)
    monkeypatch.delenv("AGENT_WEB_AUTH", raising=False)
    auth._failures.clear()
    return TestClient(app, client=("203.0.113.9", 5000))


def test_api_needs_the_token(client):
    assert client.get("/api/auth/check").status_code == 401
    assert client.get("/api/auth/check", headers={"Authorization": "Bearer wrong"}).status_code == 401
    token = auth.get_token()
    assert client.get("/api/auth/check", headers={"Authorization": f"Bearer {token}"}).json() == {"ok": True}
    assert client.get("/api/auth/check", headers={"X-Agent-Token": token}).status_code == 200


def test_static_gui_and_file_capability_urls_stay_open(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/files/not-a-real-token").status_code == 404        # past the gate: a 404, not a 401


def test_token_file_is_created_private_and_stable(client, tmp_path):
    first = auth.get_token()
    assert auth.get_token() == first and len(first) >= 32
    assert oct((tmp_path / "web-token").stat().st_mode & 0o777) == "0o600"


def test_env_token_overrides_and_auth_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("AGENT_WEB_TOKEN", "from-env")
    assert client.get("/api/auth/check", headers={"Authorization": "Bearer from-env"}).status_code == 200
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    assert client.get("/api/auth/check").status_code == 200


def test_repeated_failures_are_rate_limited(client):
    for _ in range(auth.MAX_FAILURES):
        assert client.get("/api/auth/check").status_code == 401
    assert client.get("/api/auth/check").status_code == 429
    assert client.get("/api/auth/check", headers={"Authorization": f"Bearer {auth.get_token()}"}).status_code == 429


def test_trusted_ip_skips_token_but_loopback_cannot_be_trusted(tmp_path, monkeypatch):
    from agent.web import auth

    path = tmp_path / "ips"
    monkeypatch.setattr(auth, "TRUSTED_PATH", path)
    assert auth.set_trusted("100.64.0.6") == {"100.64.0.6"}
    assert auth.trusted_ips() == {"100.64.0.6"}
    assert auth.set_trusted("100.64.0.6", on=False) == set()
    for bad in ("127.0.0.1", "0.0.0.0", "::1", "nonsense"):
        try:
            auth.set_trusted(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_middleware_lets_a_trusted_peer_in_without_token(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agent.web import auth
    from agent.web.app import app

    monkeypatch.setenv("AGENT_WEB_TOKEN", "secret")
    path = tmp_path / "ips"
    monkeypatch.setattr(auth, "TRUSTED_PATH", path)
    c = TestClient(app)  # peer address is "testclient"
    assert c.get("/api/auth/check").status_code == 401
    path.write_text("testclient\n")
    assert c.get("/api/auth/check").status_code == 200
