"""`/context`: what the NEXT prompt is made of, with a token estimate per part and its share of the budget.

Built from the same inputs `agent.memory.recall.build_context` reads (persona, personal facts, semantic recalls,
similar skills, the running summary, working notes, the history that still fits, plan, project notes), but it never
mutates anything: no compaction is written, nothing is stored, and the only model call is the one embedding of the
last user line for the recall preview (skipped when there is none or it fails). Works offline with fakes.

`collect(state)` returns `list[Part]`; `render(parts, ...)` turns them into a rich renderable; `inspector(state)`
returns the zero-argument callable `NightageApp(context_provider=...)` wants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group
from rich.text import Text

from agent.llm import EmbeddingUnavailable
from agent.ui import facts as F

CHARS_PER_TOKEN = 3.2      # same rule of thumb as agent.memory.recall's budget comment
PREVIEW = 3                # detail lines shown per part


def est_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN + 0.999) if chars > 0 else 0


@dataclass
class Part:
    name: str
    chars: int = 0
    note: str = ""                                  # short remark ("3 recalled, min similarity 0.62")
    lines: list[str] = field(default_factory=list)  # preview of the content, one per item

    @property
    def tokens(self) -> int:
        return est_tokens(self.chars)


def _clip(text: str, n: int = 90) -> str:
    one = " ".join(str(text).split())
    return one if len(one) <= n else one[: n - 3] + "..."


def _recalls(state: Any, query: str) -> tuple[list[tuple[str, str, float]], str]:
    """(text, kind, similarity) for the query, mirroring recall._semantic_hits' cut-offs. Never raises."""
    vectors, client, config = state.vectors, state.client, state.config
    try:
        if vectors is None or not vectors.available or not query.strip():
            return [], "no recall" if not query.strip() else "recall index not available"
        k = int(config["memory"]["recall_k"])
        emb = client.embed([query], model=config["models"]["embed"])[0]
        hits = vectors.search(emb, k=k, kinds=["fact", "message", "web"])
    except EmbeddingUnavailable:
        return [], "semantic memory is off (embedding model not available): keyword recall only"
    except Exception as exc:  # noqa: BLE001
        return [], f"recall unavailable ({type(exc).__name__})"
    from agent.memory import recall as R

    min_sim = float((config.get("memory") or {}).get("recall_min_similarity", 0.0) or 0.0)
    out: list[tuple[str, str, float]] = []
    web = 0
    for text, kind, ref, dist in hits:
        sim = 1.0 - dist * dist / 2.0
        if min_sim and sim < min_sim:
            continue
        if kind == "message" and len(text.strip()) < R.MIN_MESSAGE_RECALL_CHARS:
            continue
        if kind == "web":
            if sim < max(min_sim, R.WEB_NOTE_MIN_SIMILARITY) or web >= 2 or R._web_note_too_old(text):
                continue
            web += 1
        out.append((text, kind, sim))
    return out, f"min similarity {min_sim:g}" if min_sim else "no similarity cut-off"


def collect(state: Any, query: str | None = None) -> list[Part]:
    """The parts of the next prompt. `query` is the line recalls are previewed for (default: the last user
    message of the session). Pure read."""
    from agent.memory import recall as R

    cfg, store, sid = state.config, state.store, state.session_id
    parts: list[Part] = []
    tool_names = [t.name for t in state.registry.list()] if state.registry is not None else []

    persona = str((cfg.get("persona") or {}).get("style") or "")
    parts.append(Part("persona", len(persona), f"{(cfg.get('persona') or {}).get('name', '')}".strip(), [_clip(persona)] if persona else []))
    instr = R._FORMATTING_GUIDANCE + ("\n\n" + R._tool_instructions(tool_names) if tool_names else "")
    parts.append(Part("system rules + tools", len(instr), f"{len(tool_names)} tools", [_clip(", ".join(tool_names), 120)] if tool_names else []))
    try:
        from agent.memory.environment import system_facts

        env = system_facts(cfg)
    except Exception:  # noqa: BLE001
        env = []
    parts.append(Part("environment", sum(len(e) + 3 for e in env), f"{len(env)} lines", [_clip(e) for e in env[:PREVIEW]]))

    compacted, through = (None, None)
    notes = None
    history: list[dict[str, Any]] = []
    fact_rows: list[dict[str, Any]] = []
    if store is not None:
        try:
            compacted, through = store.get_session_compaction(sid)
        except Exception:  # noqa: BLE001
            pass
        try:
            notes = store.get_working_notes(sid)
            fact_rows = store.get_active_facts(kind="personal")
            history = [m for m in store.get_messages(sid) if m["role"] not in R.RENDER_ONLY_ROLES]
        except Exception:  # noqa: BLE001
            pass

    facts = [f["content"] for f in fact_rows]
    stale = sum(1 for f in fact_rows if f.get("confidence") == "assumed")
    parts.append(Part("facts about you", sum(len(f) + 3 for f in facts), f"{len(facts)} personal" + (f", {stale} assumed" if stale else ""),
                      [_clip(f) for f in facts[:PREVIEW]]))
    parts.append(Part("session notes", len(notes or ""), "" if notes else "none", [_clip(n) for n in (notes or "").splitlines()[:PREVIEW]]))
    parts.append(Part("earlier summary", len(compacted or ""), "" if compacted else "none yet", [_clip(compacted)] if compacted else []))

    q = query if query is not None else next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    hits, why = _recalls(state, q)
    recall_text = "".join(f"- {t}\n" for t, _k, _s in hits)
    parts.append(Part("recalled memories", len(recall_text), f"{len(hits)} kept ({why})" + (f" for: {_clip(q, 40)}" if q else ""),
                      [f"{s:.2f} {k}: {_clip(t, 70)}" for t, k, s in hits[:PREVIEW]]))
    skills: list[dict] = []
    try:
        from agent.memory.skills import find_similar_skill

        skills = find_similar_skill(store, q) if (store is not None and q) else []
    except Exception:  # noqa: BLE001
        skills = []
    parts.append(Part("similar past skills", sum(len(str(s.get("problem_description", ""))) + 40 for s in skills), f"{len(skills)}",
                      [_clip(s.get("title", "")) for s in skills[:PREVIEW]]))

    plan = getattr(state.plan_holder, "plan", None)
    plan_text = plan.system_message() if plan is not None else ""
    parts.append(Part("plan", len(plan_text), "none" if not plan_text else f"{len(getattr(plan, 'items', []))} checks",
                      [_clip(ln) for ln in (plan.lines()[:PREVIEW] if plan is not None else [])]))
    proj_text, proj_key = "", getattr(state.active_project, "key", None)
    if proj_key and state.project_memory is not None:
        try:
            proj_text = state.project_memory.system_message(proj_key) or ""
        except Exception:  # noqa: BLE001
            proj_text = ""
    parts.append(Part("project notes", len(proj_text), _clip(str(proj_key), 40) if proj_key else "no active project",
                      [_clip(ln) for ln in proj_text.splitlines()[1:1 + PREVIEW]]))

    budget = R.history_char_budget(cfg)
    fixed = sum(p.chars for p in parts)
    used, kept = fixed, []
    for m in reversed(history):
        if used + len(m["content"]) > budget:
            break
        kept.append(m)
        used += len(m["content"])
    kept.reverse()
    while kept and kept[0]["role"] == "tool":
        kept.pop(0)
    dropped = len(history) - len(kept)
    tool_msgs = [m for m in kept if m["role"] == "tool"]
    chat_msgs = [m for m in kept if m["role"] != "tool"]
    parts.append(Part("conversation", sum(len(m["content"]) for m in chat_msgs),
                      f"{len(chat_msgs)} messages" + (f", {dropped} older would be summarised" if dropped else ""),
                      [f"{m['role']}: {_clip(m['content'], 70)}" for m in chat_msgs[-PREVIEW:]]))
    parts.append(Part("recent tool results", sum(len(m["content"]) for m in tool_msgs), f"{len(tool_msgs)} kept",
                      [_clip(m["content"], 80) for m in tool_msgs[-PREVIEW:]]))
    return parts


def render(parts: list[Part], *, model: str = "", limit: int = 0, explain: dict[str, Any] | None = None,
           plain: bool = False, width: int = 80, history_budget_chars: int | None = None) -> Group:
    """The inspector text. `limit` is the context window (tokens) percentages are taken from."""
    total = sum(p.tokens for p in parts)
    dim = "" if plain else "dim"
    rows: list[Text] = []
    head = f"next prompt ~ {total:,} tokens"
    if limit:
        head += f" of {limit:,} ({round(100 * total / limit)}%)"
    rows.append(Text(head + (f"   model {model}" if model else ""), style="" if plain else "bold"))
    for p in parts:
        if p.chars == 0 and not p.note:
            continue
        pct = f"{100 * p.tokens / limit:4.1f}%" if limit else "     "
        t = Text(f"  {p.name:<22} {p.tokens:>7,} tok {pct}")
        if p.note:
            t.append("  " + p.note, style=dim)
        rows.append(t)
        for ln in p.lines[:PREVIEW]:
            rows.append(Text(f"      {ln}"[: max(20, width - 1)], style=dim))
    if explain:
        ov = explain.get("override")
        rows.append(Text(""))
        rows.append(Text("context budget", style="" if plain else "bold"))
        rows.append(Text(f"  computed {explain.get('computed'):,}  override {'none' if ov is None else f'{ov:,}'}  "
                         f"final {explain.get('final'):,}  ceiling {explain.get('ceiling'):,}  source {explain.get('source')}"
                         if all(isinstance(explain.get(k), int) for k in ("computed", "final", "ceiling")) else
                         f"  {explain}"))
    if history_budget_chars:
        rows.append(Text(f"  history keeps ~{history_budget_chars:,} characters (~{est_tokens(history_budget_chars):,} tokens)", style=dim))
    rows.append(Text("  estimates (~3.2 chars/token); the model reports exact counts after each turn (see /usage)", style=dim))
    return Group(*rows)


def inspector(state: Any, *, plain: bool = False, width: Callable[[], int] | None = None,
              model: Callable[[], str] | None = None) -> Callable[[], Group]:
    """context_provider for NightageApp. Reads `state` live on every call, never raises."""

    def provide() -> Any:
        try:
            from agent.memory import recall as R

            m = model() if model else state.model
            limit = F.ctx_limit(state.config, m)
            return render(collect(state), model=m, limit=limit, explain=F.budget_explain(state.config, m) or None, plain=plain,
                          width=width() if width else 80, history_budget_chars=R.history_char_budget(state.config))
        except Exception as exc:  # noqa: BLE001
            return Text(f"/context failed: {exc}")

    return provide
