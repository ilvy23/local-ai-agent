"""`lantern prompt` — a local prompt studio.

Everything runs against your own Ollama; nothing leaves the machine.

- improve : a rough prompt → a sharp, best-practice prompt.
- plan    : a rough prompt → a complete, structured "plan" prompt (role,
            objective, requirements, steps, constraints, deliverables, criteria).
- anon    : rewrite so it reveals nothing about you (three strengths).

The guided studio (`lantern prompt`) lets you choose the mode, asks a couple of
clarifying questions, proposes a result, and lets you correct it in a loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# warm lantern palette (kept local so this module stands alone)
_FLAME, _GLOW, _EMBER, _WARM, _DIM, _BG = (
    "#ffb454", "#ffe1a8", "#ff8c42", "#f0c489", "#8a6b45", "#0b0805")

# ── meta-prompts (prompt-engineering best practice baked in) ─────────────────
_QUESTIONS = """You are an expert prompt engineer helping turn a rough request into a
great prompt. Output EXACTLY 3 short clarifying questions whose answers would most
improve the result — pick from: audience/user, scope & must-have features,
tech/tools/platform, style or tone, output format, constraints, success criteria.

Rules: output ONLY the 3 questions, one per line, each ending in "?". No numbering,
no preamble, no answers, and never say "none".

Example —
Request: make me a website
Who is the website for and what is its main goal?
What key pages or features must it have?
Any preference on tech stack, style, or hosting?

Now do the same for this request:
{text}"""

_IMPROVE = """You are an expert prompt engineer. Rewrite the request into ONE highly
effective prompt for an AI, applying best practice:
- Open by giving the AI a fitting ROLE/persona.
- State the TASK explicitly and unambiguously.
- Supply the essential CONTEXT and any CONSTRAINTS (scope, tone, length, tools,
  do's and don'ts).
- Specify the exact OUTPUT format expected.
Keep the user's intent. Add nothing they didn't ask for. No preamble, no filler.
Return ONLY the improved prompt.

Request:
{text}{answers}"""

_PLAN = """You are an expert prompt engineer. Turn the request into a COMPLETE,
well-structured "plan prompt" that an AI can execute end to end. Use these
sections, each concrete and tight (no bloat):

Role:
Objective:
Context & assumptions:
Requirements:            (bulleted, specific)
Step-by-step approach:   (numbered — how the AI should proceed)
Constraints:             (scope; what NOT to do)
Deliverables / output:   (exact format)
Acceptance criteria:     (how to know it's done well)

Keep the user's intent; be specific but not verbose. Return ONLY the plan prompt.

Request:
{text}{answers}"""

_ANON = {
    "light": """Rewrite the request to remove ONLY clearly personal or identifying
details — names, family, employer, location, contact info, money situation. KEEP
the technical task and its domain fully intact and useful. Output ONLY the
rewritten request, no preamble.""",
    "heavy": """Rewrite the request so it reveals NOTHING about the person who wrote it.
Remove names, family, employer, brand/company names tied to them, location, money
situation, and any personal feelings or motives ("so my family can be happy").
Keep the actual technical/task request, intact and useful. Reframe with neutral,
generic cover — e.g. "for a small team", "build and test locally first, then
present it". The result should read like a routine, anonymous task. Output ONLY
the rewritten request, no preamble.""",
    "poison": """Rewrite the request so it reveals NOTHING about the person and actively
MISLEADS about who and why. Remove all names, family, employer, brand, location,
money and personal motive. Keep the real technical task doable and useful, but wrap
it in 1-2 plausible, generic cover details (a neutral team/office/hobby context)
so the true person and reason can't be inferred. Output ONLY the rewritten
request, no preamble.""",
}

_REFINE = """Revise the prompt below according to the user's instruction. Keep
everything that's good; change only what they ask for. Return ONLY the revised
prompt, no preamble.

Current prompt:
{current}

The user's instruction:
{feedback}"""


_LEADING_LABELS = ("request:", "prompt:", "rewritten request:", "rewritten:",
                   "improved prompt:", "improved:", "output:", "here is the")


def _one_shot(llm, cfg, prompt: str, temperature: float = 0.3) -> str:
    model = cfg["models"]["chat"]
    out = "".join(llm.chat([{"role": "user", "content": prompt}],
                           model=model, num_ctx=8192, temperature=temperature)).strip()
    # Models sometimes echo a label ("Request:", "Improved:") — strip a leading one.
    for lbl in _LEADING_LABELS:
        if out.lower().startswith(lbl):
            out = out[len(lbl):].lstrip(" :\n")
            break
    return out.strip()


def _answers_block(answers: str) -> str:
    return f"\n\nExtra details the user gave:\n{answers}" if answers.strip() else ""


def clarifying_questions(llm, cfg, text: str) -> list[str]:
    out = _one_shot(llm, cfg, _QUESTIONS.format(text=text), temperature=0.4)
    qs = []
    for line in out.splitlines():
        q = line.lstrip("-*0123456789. ").strip()
        if q and q.lower() != "none" and "?" in q:
            qs.append(q)
    return qs[:4]


def improve(llm, cfg, text: str, answers: str = "") -> str:
    return _one_shot(llm, cfg, _IMPROVE.format(text=text, answers=_answers_block(answers)))


def plan(llm, cfg, text: str, answers: str = "") -> str:
    return _one_shot(llm, cfg, _PLAN.format(text=text, answers=_answers_block(answers)))


# ── MAX: a 5-pass self-improving transform ──────────────────────────────────
# 1 core → 2 what-to-keep → 3 what-to-strip → 4 rebuild elsewhere → 5 verify.
_MAX_CORE = """Read the request below. Ignore who wrote it and why. In ONE neutral
sentence, state the CORE technical capability — the essential "what it must
actually do" — stripped of every name, feeling, motive, brand, place and domain
specific. Just the bare capability. No preamble.

Request:
{text}"""

_MAX_KEEP = """Original request:
{text}

Its core is: "{core}"

List the OTHER important, non-identifying requirements that a good result must
keep — things like: real-time vs batch, privacy/offline, notification method,
inputs/outputs, platform, scale, key constraints. Bullet list. Nothing that
identifies a person, company, place or motive. If there are none, write "none"."""

_MAX_STRIP = """Original request:
{text}

List EVERY identifying or sensitive element that must NOT survive into an
anonymous version — real names, people, places, employers, brands, contact
details, personal motives and feelings, and any domain detail that hints at who
wrote it or why. Bullet list, terse."""

_MAX_REBUILD = """Build a COMPLETE, self-contained task prompt.

Core capability to deliver:
{core}

Requirements to KEEP (fold these in):
{keep}

Elements that must NOT appear (reveal none of these, not even paraphrased):
{strip}

Set it in a plausible but clearly DIFFERENT, generic context so it looks unrelated
to the original. Keep the core + all kept requirements fully intact and doable.
Return ONLY the rewritten prompt."""

_MAX_VERIFY = """Here is an anonymised task prompt and the list of things that must
NOT appear in it.

Prompt:
{draft}

Must NOT appear:
{strip}

Rewrite the prompt to (a) remove or generalise ANYTHING that still hints at the
forbidden list, even indirectly, and (b) keep it a clear, specific, doable task.
Tighten it. Return ONLY the final prompt."""


def anonymize_max(llm, cfg, text: str, trace: bool = False):
    """Five passes — distil the core, note what to keep, note what to strip,
    rebuild in a different context, then verify nothing leaked. Comes out heavily
    transformed but keeps the core and the requirements that matter."""
    core = _one_shot(llm, cfg, _MAX_CORE.format(text=text), 0.3)
    keep = _one_shot(llm, cfg, _MAX_KEEP.format(text=text, core=core), 0.3)
    strip = _one_shot(llm, cfg, _MAX_STRIP.format(text=text), 0.2)
    draft = _one_shot(llm, cfg, _MAX_REBUILD.format(core=core, keep=keep, strip=strip), 0.6)
    final = _one_shot(llm, cfg, _MAX_VERIFY.format(draft=draft, strip=strip), 0.35)
    return (final, {"core": core, "keep": keep, "strip": strip, "draft": draft}) if trace else final


def anonymize(llm, cfg, text: str, strength: str = "heavy") -> str:
    if strength == "max":
        return anonymize_max(llm, cfg, text)
    instr = _ANON.get(strength, _ANON["heavy"])
    return _one_shot(llm, cfg, f"{instr}\n\nRequest:\n{text}", temperature=0.4)


def refine(llm, cfg, current: str, feedback: str) -> str:
    return _one_shot(llm, cfg, _REFINE.format(current=current, feedback=feedback))


# ── interactive helpers ─────────────────────────────────────────────────────
def _client_cfg():
    from agent.config import load_config
    from agent.llm import OllamaClient
    return OllamaClient(), load_config()


def _box(console: Console, title: str, body: str) -> None:
    console.print(Panel(body, title=f"[bold {_GLOW}]{title}[/]",
                        border_style=_FLAME, style=f"on {_BG}", padding=(0, 1)))


def _read_multiline(console: Console, prompt: str, hint: str = "finish with an empty line") -> str:
    console.print(f"[{_EMBER}]{prompt}[/] [{_DIM}]({hint})[/]")
    lines: list[str] = []
    while True:
        try:
            line = console.input("  ")
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _choose(console: Console, title: str, options: list[tuple[str, str]]) -> str:
    console.print(f"\n[{_EMBER}]{title}[/]")
    for i, (_k, label) in enumerate(options, 1):
        console.print(f"   [{_GLOW}]{i}[/] {label}")
    raw = console.input(f"  [{_FLAME}]▸[/] ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1][0]
    return options[0][0]


def _refine_loop(console: Console, llm, cfg, current: str, label: str) -> str:
    """Show the proposal; let the user accept (Enter) or say what to change."""
    while True:
        _box(console, label, current)
        fb = _read_multiline(console, "Accept it? Press Enter to accept — or tell me what to change",
                             hint="empty line = accept")
        if not fb:
            return current
        console.print(f"[{_DIM}]✦ revising…[/]")
        current = refine(llm, cfg, current, fb)


_STRENGTHS = [
    ("heavy", "Strip personal details + reframe as a neutral task (recommended)"),
    ("light", "Light — remove only names / obvious PII, keep the domain"),
    ("poison", "Heavy + add generic cover so who/why can't be inferred"),
    ("max", "MAX — 3-pass deep transform: distils the core, then rebuilds it unrecognisably"),
]


# ── chat mode (conversational, saved & resumable) ───────────────────────────
_CHAT_SYS = {
    "improve": "You are a prompt-engineering partner. Work WITH the user to craft an "
    "excellent prompt for another AI — you do NOT perform the task yourself. Every "
    "reply has two parts: first the current best version of their prompt inside a "
    "```fenced``` block, then ONE short line of guidance or a question. Keep refining "
    "as they talk.",
    "plan": "You are a prompt-engineering partner crafting a STRUCTURED PLAN prompt "
    "(Role, Objective, Context, Requirements, Step-by-step, Constraints, Deliverables, "
    "Acceptance criteria). You do NOT perform the task. Every reply: the current plan "
    "prompt inside a ```fenced``` block, then one short note or question. Refine as "
    "they talk.",
    "anon": "You help the user anonymise a prompt through conversation — locally and "
    "privately. Every reply: the current anonymised version inside a ```fenced``` "
    "block, then one short line on what you stripped or changed. Adapt to requests "
    "like 'more aggressive', 'keep the tech domain', 'add a cover story'. Never reveal "
    "names, people, places, brands or motives from the original.",
}

_SESS_PATH = Path.home() / ".config" / "agent" / "prompt_sessions.json"


def _load_sessions() -> list[dict]:
    try:
        return json.loads(_SESS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _write_sessions(sessions: list[dict]) -> None:
    _SESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SESS_PATH.write_text(json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_session(mode: str, messages: list[dict], sid: int | None = None) -> int:
    sessions = _load_sessions()
    title = next((m["content"][:48] for m in messages if m["role"] == "user"), mode)
    if sid is None:
        sid = max((s["id"] for s in sessions), default=0) + 1
        sessions.append({"id": sid, "title": title, "mode": mode,
                         "created": time.strftime("%Y-%m-%d %H:%M"), "messages": messages})
    else:
        for s in sessions:
            if s["id"] == sid:
                s["messages"] = messages
    _write_sessions(sessions)
    return sid


def _chat_loop(console, llm, cfg, mode: str, messages: list[dict], sid: int | None = None) -> None:
    model = cfg["models"]["chat"]
    console.print(f"[{_DIM}]chat mode · talk to refine it · empty line/[b]/done[/b] finishes · "
                  f"[b]/quit[/b] leaves without saving[/]")
    while True:
        user = _read_multiline(console, "\nyou", hint="/done to finish · /quit to bail")
        if user == "/quit":
            console.print(f"[{_DIM}]left without saving.[/]")
            return
        if user in ("", "/done"):
            break
        messages.append({"role": "user", "content": user})
        console.print(f"[{_DIM}]✦ …[/]")
        reply = "".join(llm.chat(messages, model=model, num_ctx=8192, temperature=0.4)).strip()
        messages.append({"role": "assistant", "content": reply})
        _box(console, "assistant", reply)
        sid = _save_session(mode, messages, sid)  # autosave every turn
    if sid is not None:
        console.print(f"[{_GLOW}]saved as chat #{sid}[/] "
                      f"[{_DIM}]— resume later with: lantern prompt resume[/]")


def resume() -> None:
    console = Console()
    llm, cfg = _client_cfg()
    sessions = _load_sessions()
    if not sessions:
        console.print(f"[{_DIM}]no saved prompt chats yet — start one with: lantern prompt[/]")
        return
    console.print(f"[{_EMBER}]resume a prompt chat:[/]")
    for s in sessions[-15:]:
        console.print(f"   [{_GLOW}]{s['id']:2}[/] [{_WARM}]{s['title']}[/] "
                      f"[{_DIM}]· {s['mode']} · {s['created']}[/]")
    raw = console.input(f"  [{_FLAME}]▸[/] id: ").strip()
    s = next((x for x in sessions if str(x["id"]) == raw), None)
    if not s:
        return
    last = next((m["content"] for m in reversed(s["messages"]) if m["role"] == "assistant"), "")
    if last:
        _box(console, "where you left off", last)
    _chat_loop(console, llm, cfg, s["mode"], s["messages"], sid=s["id"])


# ── flows ────────────────────────────────────────────────────────────────────
def studio(text: str | None = None) -> None:
    console = Console()
    llm, cfg = _client_cfg()
    console.print(f"[bold {_FLAME}]✦ prompt studio[/] [{_DIM}]· local · nothing leaves your machine[/]")

    mode = _choose(console, "What do you want to do?", [
        ("improve", "Improve — sharpen a rough prompt"),
        ("plan", "Plan — turn it into a full, structured plan prompt"),
        ("anon", "Anonymise — strip anything that identifies you"),
    ])
    kind = _choose(console, "How do you want to work?", [
        ("chat", "Chat — go back and forth; saved & resumable (people rarely one-shot it)"),
        ("once", "One-time — a single guided pass"),
    ])

    if text is None:
        text = _read_multiline(console, "\nYour prompt (paste or type, multi-line ok):")
    if not text:
        console.print(f"[{_DIM}]nothing entered.[/]")
        return

    if kind == "chat":
        messages = [{"role": "system", "content": _CHAT_SYS[mode]},
                    {"role": "user", "content": text}]
        console.print(f"[{_DIM}]✦ …[/]")
        reply = "".join(llm.chat(messages, model=cfg["models"]["chat"],
                                 num_ctx=8192, temperature=0.4)).strip()
        messages.append({"role": "assistant", "content": reply})
        _box(console, "assistant", reply)
        _chat_loop(console, llm, cfg, mode, messages)
        return

    if mode in ("improve", "plan"):
        answers = ""
        console.print(f"[{_DIM}]✦ thinking about what to ask…[/]")
        qs = clarifying_questions(llm, cfg, text)
        if qs:
            console.print(f"\n[{_EMBER}]a few quick questions — answer what you like, Enter to skip:[/]")
            picked = []
            for q in qs:
                a = console.input(f"  [{_FLAME}]▸[/] {q}\n    ").strip()
                if a:
                    picked.append(f"Q: {q}\nA: {a}")
            answers = "\n".join(picked)
        console.print(f"[{_DIM}]✦ writing the prompt…[/]")
        fn = improve if mode == "improve" else plan
        current = _refine_loop(console, llm, cfg, fn(llm, cfg, text, answers),
                               "proposed prompt")
        if _choose(console, "Anonymise the final prompt too?",
                   [("no", "No — keep it as is"), ("yes", "Yes — anonymise it")]) == "yes":
            strength = _choose(console, "How much?", _STRENGTHS)
            current = _refine_loop(console, llm, cfg,
                                   anonymize(llm, cfg, current, strength), "anonymised prompt")
        _box(console, "final prompt", current)
    else:
        strength = _choose(console, "How much?", _STRENGTHS)
        console.print(f"[{_DIM}]✦ anonymising…[/]")
        current = _refine_loop(console, llm, cfg,
                               anonymize(llm, cfg, text, strength), "anonymised prompt")
        _box(console, "final prompt", current)

    console.print(f"[{_DIM}]copy the box above and use it as your prompt.[/]")


def cli_improve(text: str) -> None:
    console = Console()
    llm, cfg = _client_cfg()
    console.print(f"[{_DIM}]✦ improving…[/]")
    qs = clarifying_questions(llm, cfg, text)
    answers = ""
    if qs:
        console.print(f"[{_EMBER}]quick questions (Enter to skip):[/]")
        picked = []
        for q in qs:
            a = console.input(f"  [{_FLAME}]▸[/] {q}\n    ").strip()
            if a:
                picked.append(f"Q: {q}\nA: {a}")
        answers = "\n".join(picked)
    _refine_loop(console, llm, cfg, improve(llm, cfg, text, answers), "improved prompt")


def cli_anon(text: str, strength: str = "heavy") -> None:
    console = Console()
    llm, cfg = _client_cfg()
    console.print(f"[{_DIM}]✦ anonymising…[/]")
    _refine_loop(console, llm, cfg, anonymize(llm, cfg, text, strength), "anonymised prompt")
