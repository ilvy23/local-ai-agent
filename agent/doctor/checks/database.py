"""Database: integrity, schema version, size, write-ahead log. Opened read-only; the doctor never creates or changes it."""

from __future__ import annotations

from agent.doctor.core import Context, check, fail, info, ok, skip, warn

MB = 1024 ** 2


def _target_version() -> int | None:
    try:
        from agent.memory import store

        for name in ("SCHEMA_VERSION", "TARGET_VERSION", "_TARGET_VERSION"):
            v = getattr(store, name, None)
            if isinstance(v, int):
                return v
        migs = getattr(store, "MIGRATIONS", None) or getattr(store, "_MIGRATIONS", None)
        if migs:
            return len(migs)
    except Exception:  # noqa: BLE001
        pass
    return None


@check("database.integrity", "database", "Database integrity", timeout=15)
def integrity(ctx: Context):
    _dir, db = ctx.p.data_paths(ctx.config)
    if not ctx.p.exists(db):
        return skip("no database yet; it is created on first use")
    size = ctx.p.file_size(db) or 0
    res = ctx.cached("db_check", lambda: ctx.p.sqlite_check(db, quick=size > 500 * MB))
    if res.get("error"):
        return fail(f"cannot open {db.name}: {res['error']}", "If the file is damaged, restore a backup or move it away (a new one is created).")
    if res.get("integrity") != "ok":
        return fail(f"SQLite reports: {str(res.get('integrity'))[:120]}", "Restore the database from a backup (`sqlite3 agent.db .recover` can save most rows).")
    return ok(f"{'quick check' if size > 500 * MB else 'integrity check'} passed")


@check("database.schema", "database", "Database schema")
def schema(ctx: Context):
    _dir, db = ctx.p.data_paths(ctx.config)
    if not ctx.p.exists(db):
        return skip("no database yet")
    res = ctx.cached("db_check", lambda: ctx.p.sqlite_check(db))
    if res.get("error"):
        return skip("database not readable (see the integrity row)")
    v, target = int(res.get("user_version") or 0), _target_version()
    text = f"schema version {v}, {len(res.get('tables') or [])} tables"
    if target is not None and v < target:
        return info(text + f"; this version of nightage upgrades it to {target} on the next start")
    if target is not None and v > target:
        return warn(text + f", but this nightage only knows up to {target}", "The database was written by a newer nightage; update before using it.")
    return ok(text)


@check("database.size", "database", "Database size and journal")
def size(ctx: Context):
    _dir, db = ctx.p.data_paths(ctx.config)
    sz = ctx.p.file_size(db)
    if sz is None:
        return skip("no database yet")
    wal = ctx.p.file_size(str(db) + "-wal") or 0
    text = f"{sz / MB:.1f} MB" + (f", WAL {wal / MB:.1f} MB" if wal else "")
    if wal > 200 * MB:
        return warn(text, "The write-ahead log is large (a long-running process holds it open): quit nightage everywhere and restart it.")
    if sz > 4 * 1024 * MB:
        return warn(text, "A very large database: search gets slower; consider pruning old sessions.")
    return ok(text)
