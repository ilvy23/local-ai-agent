"""Range streaming + content types for /api/files, and send_album (planning, approval text, ordering, caps, persistence)."""

from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import repoaccess  # noqa: E402
from agent.chat.turn import plain_description  # noqa: E402
from agent.config import DEFAULT_CONFIG  # noqa: E402
from agent.memory.store import Store  # noqa: E402
from agent.safety import RiskLevel  # noqa: E402
from agent.tools import send_album as sa  # noqa: E402
from agent.tui import _classify_call  # noqa: E402
from agent.web import deps  # noqa: E402
from agent.web.app import app  # noqa: E402


class Console:
    def __init__(self):
        self.events = []

    def emit_event(self, event_type, **fields):
        self.events.append({"type": event_type, **fields})

    def print(self, *a, **k):
        pass


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WEB_AUTH", "off")
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["data"]["db_path"] = str(tmp_path / "a.db")
    monkeypatch.setattr(deps, "_config", config)
    store = Store(config["data"]["db_path"])
    sid = store.create_session("s")
    console = Console()
    return TestClient(app), tmp_path, store, sid, console, sa.make_send_album_tool(store, sid, console)


DATA = bytes(range(256)) * 4        # 1024 bytes


def share(store, sid, tmp, name, disposition="inline", data=DATA):
    f = tmp / name
    f.write_bytes(data)
    return f"/api/files/{store.add_file_share(sid, str(f), disposition=disposition)}"


def test_range_206_headers_and_body(env):
    tc, tmp, store, sid, *_ = env
    url = share(store, sid, tmp, "a.mp3")
    r = tc.get(url, headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == DATA[10:20]
    assert r.headers["content-range"] == "bytes 10-19/1024" and r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-type"] == "audio/mpeg" and r.headers["x-content-type-options"] == "nosniff"
    assert tc.get(url).headers["accept-ranges"] == "bytes"


def test_range_open_ended_and_suffix(env):
    tc, tmp, store, sid, *_ = env
    url = share(store, sid, tmp, "a.mp4")
    assert tc.get(url, headers={"Range": "bytes=1000-"}).content == DATA[1000:]
    r = tc.get(url, headers={"Range": "bytes=-24"})                  # suffix: the last 24 bytes
    assert r.status_code == 206 and r.content == DATA[-24:] and r.headers["content-range"] == "bytes 1000-1023/1024"


@pytest.mark.parametrize("rng", ["bytes=5000-6000", "bytes=2000-", "bytes=20-10", "items=0-5"])
def test_range_invalid_is_416(env, rng):
    tc, tmp, store, sid, *_ = env
    r = tc.get(share(store, sid, tmp, "a.webm"), headers={"Range": rng})
    assert r.status_code == 416 and r.headers["content-range"] == "bytes */1024"


@pytest.mark.parametrize("name,mime", [("a.mp4", "video/mp4"), ("a.webm", "video/webm"), ("a.mkv", "video/x-matroska"),
                                       ("a.mp3", "audio/mpeg"), ("a.flac", "audio/flac"), ("a.ogg", "audio/ogg"),
                                       ("a.opus", "audio/ogg"), ("a.wav", "audio/wav"), ("a.m4a", "audio/mp4"), ("a.png", "image/png")])
def test_inline_content_types(env, name, mime):
    tc, tmp, store, sid, *_ = env
    r = tc.get(share(store, sid, tmp, name))
    assert r.headers["content-type"].split(";")[0] == mime and r.headers["content-disposition"].startswith("inline")


@pytest.mark.parametrize("name", ["x.html", "x.svg", "x.js", "x.xyz", "x.txt"])
def test_inline_never_serves_active_or_unknown_types(env, name):
    tc, tmp, store, sid, *_ = env
    r = tc.get(share(store, sid, tmp, name))                         # asked for inline, must still be a download
    assert r.headers["content-disposition"].startswith("attachment") and r.headers["content-type"].startswith("application/octet-stream")
    assert r.headers["x-content-type-options"] == "nosniff"


def test_attachment_media_stays_octet_stream_and_unknown_token_404(env):
    tc, tmp, store, sid, *_ = env
    r = tc.get(share(store, sid, tmp, "a.mp3", disposition="attachment"))
    assert r.headers["content-type"].startswith("application/octet-stream") and r.headers["content-disposition"].startswith("attachment")
    assert tc.get("/api/files/not-a-token").status_code == 404
    assert tc.get("/api/files/").status_code in (404, 405)           # no listing route


def test_tokens_are_long_and_unique(env):
    _, tmp, store, sid, *_ = env
    (tmp / "f.mp3").write_bytes(b"x")
    toks = {store.add_file_share(sid, str(tmp / "f.mp3")) for _ in range(50)}
    assert len(toks) == 50 and all(len(t) >= 43 for t in toks)       # token_urlsafe(32) = 256 bits


def make_music(tmp, names, sub="album"):
    d = tmp / sub
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"\xff\xfb" + b"0" * 10)
    return d


def test_album_natural_order_skips_junk_and_hidden(env):
    tc, tmp, store, sid, console, tool = env
    d = make_music(tmp, ["10 - ten.mp3", "2 - two.mp3", "1 - one.mp3", "notes.txt", ".hidden.mp3", "cover.jpg", "x.html"])
    out = tool.handler(path=str(d))
    assert out.startswith("Album sent") and len(console.events) == 1
    ev = console.events[0]
    assert ev["type"] == "file_ready" and ev["kind"] == "album" and ev["album_kind"] == "audio" and ev["count"] == 3
    assert [i["filename"] for i in ev["items"]] == ["1 - one.mp3", "2 - two.mp3", "10 - ten.mp3"]
    assert set(ev["items"][0]) >= {"token", "url", "title", "size", "mime"} and ev["items"][0]["mime"] == "audio/mpeg"
    assert ev["cover_url"].startswith("/api/files/")                  # cover.jpg picked up
    assert tc.get(ev["items"][0]["url"], headers={"Range": "bytes=0-1"}).status_code == 206
    assert str(tmp) not in json.dumps(ev)                             # no filesystem path leaks into the event


def test_album_explicit_list_keeps_order_and_majority_kind(env):
    _, tmp, store, sid, console, tool = env
    d = make_music(tmp, ["b.mp3", "a.mp3", "p.png"])
    tool.handler(paths=[str(d / "b.mp3"), str(d / "a.mp3"), str(d / "p.png")])
    ev = console.events[0]
    assert [i["filename"] for i in ev["items"]] == ["b.mp3", "a.mp3"]


def test_photo_and_video_albums(env):
    _, tmp, store, sid, console, tool = env
    d = make_music(tmp, ["1.jpg", "2.png", "3.webp", "s.svg"], "photos")
    tool.handler(path=str(d))
    assert console.events[-1]["album_kind"] == "image" and console.events[-1]["count"] == 3 and "cover_url" not in console.events[-1]
    v = make_music(tmp, ["e1.mp4", "e2.webm"], "vids")
    tool.handler(path=str(v), title="Season 1")
    assert console.events[-1]["album_kind"] == "video" and console.events[-1]["name"] == "Season 1"


def test_album_cap_and_errors(env, monkeypatch):
    _, tmp, store, sid, console, tool = env
    d = make_music(tmp, [f"{i:03}.mp3" for i in range(sa.MAX_ITEMS + 1)])
    assert "limit" in tool.handler(path=str(d)) and not console.events
    ok = make_music(tmp, [f"{i:03}.mp3" for i in range(sa.MAX_ITEMS)], "ok")
    assert tool.handler(path=str(ok)).startswith("Album sent")
    empty = make_music(tmp, ["a.txt"], "empty")
    assert tool.handler(path=str(empty)).startswith("Error")
    assert tool.handler(path=str(tmp / "nope")).startswith("Error") and tool.handler().startswith("Error")


def test_album_symlink_out_of_folder_is_skipped(env):
    _, tmp, store, sid, console, tool = env
    d = make_music(tmp, ["a.mp3"])
    secret = tmp / "secret.mp3"
    secret.write_bytes(b"top secret")
    (d / "link.mp3").symlink_to(secret)
    tool.handler(path=str(d))
    assert [i["filename"] for i in console.events[0]["items"]] == ["a.mp3"]


def test_approval_summary_and_risk(env):
    _, tmp, store, sid, console, tool = env
    d = make_music(tmp, ["a.mp3", "b.mp3", "c.txt"])
    text = plain_description("send_album", {"path": str(d)})
    assert "2 audio files" in text and "album" in text and "24 B" in text
    assert "not a folder" in plain_description("send_album", {"path": str(tmp / "x")})
    assert _classify_call(tool, {"path": str(d)}, deps._config) is RiskLevel.CAUTION
    assert repoaccess.allows("send_album", {"path": str(d)}, True) is False      # no folder grant lifts it
    assert deps.build_registry(store, None, None, deps._config, sid).get("send_album") is not None


def test_group_persists_and_shared_files_collapses_it(env):
    tc, tmp, store, sid, console, tool = env
    d = make_music(tmp, ["1.mp3", "2.mp3"])
    tool.handler(path=str(d))
    ev = console.events[0]
    from agent.web.routes.chat import _persisting_emit

    import asyncio

    async def noop(e): ...

    asyncio.run(_persisting_emit(store, sid, noop)(ev))
    stored = [json.loads(m["content"]) for m in store.get_messages(sid) if m["role"] == "file_card"]
    assert stored[0]["kind"] == "album" and len(stored[0]["items"]) == 2          # redrawn after reload
    rows = tc.get("/api/apps/shared-files").json()
    assert len(rows) == 1 and rows[0]["kind"] == "album" and rows[0]["count"] == 2 and rows[0]["name"] == "album"
    assert [i["filename"] for i in rows[0]["items"]] == ["1.mp3", "2.mp3"] and "path" not in json.dumps(rows)
