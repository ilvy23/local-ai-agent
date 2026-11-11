"""The morning report: per prompt what was measured (status, attempts, minutes, failure kinds, the last note), the checkpoints, the lessons and the
unfinished work. Built from the governor's state and git only - nothing in it is the model's own claim."""
from __future__ import annotations

from agent.night import attemptlog, gitbox


def triage(night) -> list[str]:
    """The prompts worth a human look first, with the reasons that were MEASURED (many attempts, guard reverts, a tournament, a discarded review, not finished) and the command
    that shows the diff. Empty when nothing stands out."""
    out = []
    prev = "night/base"
    scored = []
    for it in night.items:
        mine = [e for e in night.events if f"[{it.name}]" in e]
        reasons = []
        if it.attempts >= 4:
            reasons.append(f"{it.attempts} attempts")
        n_guard = sum(1 for e in mine if "guard:" in e)
        if n_guard:
            reasons.append(f"{n_guard} guard revert(s) (weakened/deleted tests or protected files)")
        if any("tournament" in e for e in mine):
            reasons.append("needed a tournament")
        if any("review discarded" in e for e in mine):
            reasons.append("its review was discarded")
        if any("rolled back" in e for e in mine):
            reasons.append("was rolled back at least once")
        if it.status != "done":
            reasons.append("NOT finished: " + (it.note or "no time left")[:80])
        tag = f"night/done-{gitbox.slug(it.name)}"
        cmd = f"git diff {prev} {tag}" if gitbox.tags(night.root, tag) else f"git diff night/base {gitbox.last_green(night.root)}"
        if gitbox.tags(night.root, tag):
            prev = tag
        scored.append((len(reasons), it.name, reasons, cmd))
    for n, name, reasons, cmd in sorted(scored, key=lambda r: -r[0]):
        if n:
            out.append(f"- {name}: " + "; ".join(reasons) + f"   ({cmd})")
    return out


def render(night, tests_ok: bool) -> str:
    lines = ["# Night report", "", f"tests at the end: {'green' if tests_ok else 'RED (no green checkpoint was better)'}",
             f"time used: {(night.clock() - night.start) / 3600:.1f} h of {(night.end - night.start) / 3600:.1f} h", "",
             "| prompt | status | attempts | minutes | failure kinds | last note |", "|---|---|---|---|---|---|"]
    for i in night.items:
        kinds = ", ".join(f"{k}x{i.kinds.count(k)}" for k in dict.fromkeys(i.kinds)) or "-"
        lines.append(f"| {i.name} | {i.status} | {i.attempts} | {i.seconds / 60:.0f} | {kinds} | {i.note[:100].replace('|', '/')} |")
    if getattr(night, "env_lines", None):
        lines += ["", "## Machine", ""] + [f"- {ln}" for ln in night.env_lines]
    rows = attemptlog.rows(night.attempts_file)
    if rows:
        lines += ["", "## Attempts (measured)", "", "| prompt | n | strategy | after | result | tests | diff | tok/s | min |", "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['prompt']} | {r['n']} | {r['strategy']} | {r.get('trigger') or '-'} | {r['kind']} | {r['passed']}/{r['failed']} | +{r['added']}/-{r['removed']} | {r['tps'] or '-'} | {r['minutes']} |"
                  for r in rows]
        per: dict[str, list[int]] = {}
        for r in rows:
            c = per.setdefault(r["strategy"], [0, 0])
            c[0] += 1
            c[1] += 1 if r.get("green") else 0
        steps, guards = sum(r.get("steps", 0) for r in rows), sum(r.get("guards", 0) for r in rows)
        if steps:
            lines += ["", f"Guards of the coding agent interfered {guards} times in {steps} model steps ({guards / steps * 100:.0f}%): "
                          f"{sum(r.get('blocked', 0) for r in rows)} blocked reads, {sum(r.get('think', 0) for r in rows)} forced thinking passes, "
                          f"{sum(r.get('aborts', 0) for r in rows)} loop aborts, {sum(r.get('caps', 0) for r in rows)} stage caps, {sum(r.get('masks', 0) for r in rows)} prompt cuts."]
        lines += ["", "Strategies: " + ", ".join(f"{s} {g}/{n} green" for s, (n, g) in per.items())]
    first = triage(night)
    if first:
        lines += ["", "## Look at these first", ""] + first
    done_tags = gitbox.tags(night.root, "night/done-")
    if done_tags:
        prev = "night/base"
        lines += ["", "## Diff per finished prompt", ""]
        for t in done_tags:
            f, a, r = gitbox.numstat(night.root, prev, t)
            lines.append(f"- {t.removeprefix('night/done-')}: {f} files, +{a}/-{r}")
            prev = t
    lines += ["", "## Checkpoints", "", "```", gitbox.head_summary(night.root), "```"]
    lessons = night.lessons._lines()
    if lessons:
        lines += ["", "## Lessons learned", ""] + lessons
    if night.flaky:
        lines += ["", "## Flaky tests (passed on a re-run)", ""] + [f"- {t}" for t in sorted(night.flaky)]
    open_items = [i for i in night.items if i.status != "done"]
    if open_items:
        lines += ["", "## Not finished", ""] + [f"- {i.name}: {i.note or 'no time left'} (attempt logs: .night/{i.name}-attempt*.log, red work: tags night/red-*)" for i in open_items]
    lines += ["", "## Events", ""] + [f"- {e}" for e in night.events]
    lines += ["", gitbox.review_text(night.root)]
    return "\n".join(lines) + "\n"
