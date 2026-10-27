"""Attachment upload route. Needs the web group."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.web.app import app  # noqa: E402
from agent.web.routes import uploads  # noqa: E402


@pytest.fixture()
def tc(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    monkeypatch.setenv("AGENT_UPLOAD_DIR", str(tmp_path / "up"))
    return TestClient(app), tmp_path


def test_a_file_is_stored_privately_under_its_session(tc):
    client, tmp = tc
    r = client.put("/api/uploads/7/flake.nix", content=b"{ description = 1; }")
    body = r.json()
    assert r.status_code == 200 and body["size"] == 20 and body["name"] == "flake.nix"
    path = tmp / "up" / "7" / body["path"].rsplit("/", 1)[1]
    assert path.read_bytes() == b"{ description = 1; }" and oct(path.stat().st_mode & 0o777) == "0o600"
    assert body["path"] == str(path)


def test_names_cannot_escape_the_upload_folder(tc):
    client, tmp = tc
    for evil in ("..%2F..%2Fetc%2Fpasswd", "..%5C..%5Cx.txt", "a%2Fb%2Fc.txt"):
        body = client.put(f"/api/uploads/1/{evil}", content=b"x").json()
        assert str(tmp / "up" / "1") in body["path"] and "/.." not in body["path"] and "\\" not in body["name"]


def test_empty_and_oversized_files_are_refused(tc, monkeypatch):
    client, tmp = tc
    assert client.put("/api/uploads/1/a.txt", content=b"").status_code == 422
    monkeypatch.setattr(uploads, "MAX_BYTES", 10)
    assert client.put("/api/uploads/1/big.bin", content=b"x" * 11).status_code == 413
    assert not any(p.name.endswith("big.bin") for p in (tmp / "up" / "1").glob("*"))
