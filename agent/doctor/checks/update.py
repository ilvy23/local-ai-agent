"""Updates: what the last (cached) check found and how this install updates. No network here: `nightage update --check` asks."""

from __future__ import annotations

from agent.doctor.core import Context, check, info, ok


@check("update.available", "update", "Updates")
def available(ctx: Context):
    from agent import brand, update

    rep = update.cached_report(ctx.config)
    inst = update.detect_install()
    where = inst.describe()
    if not update.enabled(ctx.config):
        return info(f"automatic update checks are off ({where}); `{brand.CMD} update --check` still works")
    if rep is None:
        return info(f"not checked yet ({where}); run `{brand.CMD} update --check`")
    lines = update.notice_lines(ctx.config)
    if lines:
        return info("; ".join(lines) + f" ({where})")
    return ok(f"up to date as of {update.iso(rep.checked_at)} ({where})")
