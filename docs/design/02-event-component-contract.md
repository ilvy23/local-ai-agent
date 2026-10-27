# 02 · Event & component contract — for the web GUI

The data the GUI receives, so mockups use **real shapes and real edge cases**. Read together with `01-product-ux-brief.md`.
Status tags: **EXISTS** (implemented today) · **NEW** (planned, will be implemented to this spec) · **OPEN** (decide while designing; tell us).
All payloads are JSON. Timestamps are ISO-8601. Strings may be German or English and can be very long.

---

## 1. Transport

- **REST** under `/api/…` for everything that is a request/response.
- **Server-Sent Events (SSE)** for a running chat turn: `POST /api/sessions/{id}/messages` with body `{"text": "...", "web_forced": false}` returns a `text/event-stream`. Each message is
  `event: <type>\ndata: {"type": "<type>", ...}\n\n`. The stream ends after a terminal event.
- **Terminal events** (the stream closes after one of these): `done`, `error`, `awaiting_approval`.
  After `awaiting_approval` the turn is *suspended*; the decision call (`POST /api/approvals/{id}/decide`) returns a **new stream** that continues the turn.
- **Auth (NEW)**: every request carries `Authorization: Bearer <token>`; missing/invalid → `401 {"detail": "…"}` → show the Connect screen.
- Order of events inside one turn is not strict; the UI must tolerate any order and duplicates (a reconnect may replay).

## 2. SSE events

### 2.1 Existing

| type | status | payload | UI meaning |
|---|---|---|---|
| `token` | EXISTS (real streaming) | `{"type":"token","text":"Hello "}` | append to the current assistant message. **After a `tool_call` or `block` event, the next tokens start a NEW message** (text before a tool and the final answer after it are separate bubbles; `done.reply` is only the final one) |
| `tool_call` | EXISTS | `{"type":"tool_call","name":"web_search","arguments":{"query":"…","num_results":5}}` | a tool started → tool activity row (running) |
| `tool_result` | EXISTS | `{"type":"tool_result","name":"web_search","result_preview":"Web search “…” on 2026-10-18: 5 sources…"}` (≤ 200 chars; shell shows `[ran: ls -la]`) | tool finished → row (done); preview collapsible |
| `awaiting_approval` | EXISTS | `{"type":"awaiting_approval","approval_id":42,"tool_name":"run_command","arguments":{"command":"rm -r build"},"risk_level":"DANGEROUS","description":"rm -r build"}` | **terminal**; show approval card |
| `file_ready` | EXISTS | `{"type":"file_ready","kind":"audio|video|image|file","url":"/api/files/<token>","filename":"a.mp3","size":"4.2 MB","subtitle_url?":"/api/files/…","subtitle_label?":"English"}` | player/download card. Album variant (send_album): `{"type":"file_ready","kind":"album","album_kind":"audio|video|image","name":"…","count":12,"size":"84 MB","cover_url?":"/api/files/…","items":[{"token","url","title","filename","size":bytes,"mime","duration?","artist?"}]}`. Also stored as a message with role `file_card` (content = this JSON) so reload redraws it |
| `image_job` | EXISTS | `{"type":"image_job","job_id":"ab12","eta_s":240,"prompt":"a lantern in rain"}` | start polling `GET /api/images/{job_id}` every few seconds → `{"status":"running|done|failed","prompt":"…","error":null,"url":null,"filename":null,"size":null}` |
| `log` | EXISTS | `{"type":"log","text":"searching for: …"}` | transient narration line (dim, auto-hide) |
| `done` | EXISTS | `{"type":"done","reply":"<full final text>"}` | **terminal**; finalize message |
| `error` | EXISTS | `{"type":"error","message":"Too many tool calls in a row. Try rephrasing."}` | **terminal**; error bubble + retry |

Risk levels: `SAFE` (never prompts) · `CAUTION` (Approve/Deny) · `DANGEROUS` (user must type `yes`) · `BLOCKED` (never runs; shown as refused: `tool_result` with preview starting `BLOCKED by safety policy: …`).

### 2.2 Events added for the GUI (`status`, `thinking`, `sources`, `verdict`, `block`, `stuck`, `context`, `verify_result`, `plan`, `verify`, `changes`, `edit` EXISTS - the last four are now emitted by the shared turn engine, see §2.3; `model` still NEW)

```jsonc
// status — what the assistant is doing while nothing can be shown yet
{"type":"status","state":"thinking|planning|compacting|searching|reading|coding|editing|verifying",
 "label":"editing","detail":"rps/game.py","since":"2026-10-18T14:03:11Z"}

// thinking — activity pulse only; the reasoning text itself is NEVER sent to the UI
{"type":"thinking","tokens":1240}

// sources — emitted after a web tool finishes; numbered consistently for the whole turn
{"type":"sources","query":"rust 1.90 release date","engine":"ddg|mojeek|marginalia|connector:wikipedia|…",
 "searched_at":"2026-10-18T14:03:20Z",
 "items":[{
   "n":1,"url":"https://blog.rust-lang.org/…","domain":"blog.rust-lang.org",
   "title":"Announcing Rust 1.90","date":"2026-09-14","date_source":"meta|jsonld|url|text|none",
   "trust":{"tier":"T1","level":"high","reasons":["official project blog","has author","has date"],"user_rule":null},
   "origin_group":"g1","duplicate_of":null,
   "status":"ok|cached|skipped|failed|timeout|blocked","status_note":"paywall|cookie wall|needs JavaScript|captcha|pdf|http 403",
   "passage":"Rust 1.90.0 was released on September 14 …","passage_chars":[1832,2104],
   "flags":["suspicious_instructions","affiliate","clickbait","typosquat","http_only","preprint","tertiary"]}]}

// verdict — computed by code after the answer; label wording is the model's
{"type":"verdict","label":"verified|likely|single_source|conflicting|low_trust|from_memory",
 "summary":"3 sources, 2 independent, newest 14 Sep 2026",
 "independent_sources":2,"newest":"2026-09-14","oldest":"2026-03-02",
 "conflicts":[{"attribute":"release date","values":[{"value":"2026-09-14","sources":[1,3],"trust":"high"},
                                                     {"value":"2026-09-12","sources":[4],"trust":"low"}],
               "preferred":"2026-09-14","why":"newest primary source"}],
 "claims":[{"text":"released on 14 September","support":[1,3],"status":"supported|unsupported|conflicting"}],
 "warnings":["figure “43 %” appears in no source"]}

// block — a visual answer element (see §4); also stored as a message with role "block" (content = this JSON)
{"type":"block","id":"b_7f3a","kind":"chart","title":"Release cadence","spec":{ … },
 "source_ids":[1,3],"provenance":"sources|computed|model","created_at":"…"}

// plan — acceptance list for big tasks (emitted by the engine: when created, and when the model said "done" with items open)
// state: created | nudge (model sent back to work, at most twice) | unconfirmed (given up: say so in the answer)
{"type":"plan","state":"created|nudge|unconfirmed","confirmed":1,"total":7,"nudges":1,
 "items":[{"id":1,"text":"tests pass","done":false}]}      // `nudges` is absent on "created"

// verify — project check before "done" is accepted (emitted by the engine; the check command goes through the normal approval)
{"type":"verify","state":"running|passed|failed|skipped|gave_up","command":"cd /p && python3 -m pytest -q -x","tries":1,"retries":2}
// `tries`/`retries` are present on running/failed/gave_up. A `status` event with state "verifying" precedes "running".

// stuck — harness reacted to a loop (the engine now reports the real count; a `status` searching event follows a lookup)
{"type":"stuck","kind":"repeat|lookup","count":2,"query":"python no module named pytest"}

// verify_result — the answer to POST /api/sessions/{id}/verify; stored as a render-only message (role "evidence")
{"type":"verify_result","overall":"supported|contradicted|unclear|nothing_to_check","searches":4,
 "claims":[{"text":"Python 3.14 starts about 40 % faster than 3.13.","status":"supported|contradicted|unclear",
            "queries":["Python 3.14 40 faster 3.13 official source","… fact check"],
            "support":["https://docs.python.org/3.14/whatsnew"],
            "against":[{"url":"https://…","attribute":"startup (%)","claim":90,"source":40}]}]}

// context — after each model call
{"type":"context","used":41200,"limit":98304,"percent":42,"tokens_per_s":36.1,"compacted":false}

// edit — one file edit, right after the tool ran (emitted by the engine; `diff` is a unified diff, cut at 2000 lines)
{"type":"edit","path":"rps/game.py","abs_path":"/home/me/rps/game.py","added":126,"removed":0,"created":true,
 "diff":"--- /dev/null\n+++ b/rps/game.py\n@@ …","diff_truncated":false}
// On the web an edit that needed approval is reported at the start of the resumed stream (the approval decision runs it).

// changes — once at the end of a turn that edited files, before `done` (emitted by the engine)
{"type":"changes","files":[{"path":"rps/game.py","added":126,"removed":0,"created":true}],"added":126,"removed":0}
// (`undo_available` is not sent: the web has no /undo yet. The terminal keeps the undo stack.)

// model — load progress / state
{"type":"model","model":"qwen3.8:27b-text","state":"loading|ready|unloaded|error","server":"dual-gpu|single","detail":"loading 62 %"}
```

### 2.3 One engine, two front ends

`agent/chat/turn.py::run_turn` is the single turn engine behind the web chat and the terminal chat (`agent chat`'s
full-screen UI); only the classic `chat --classic` loop is separate. What the engine emits on top of the events above:

- Shared gates, switched with `chat.gates.{plan,goal,step_extension,changes,verify}` in the config. Web defaults: `plan`,
  `goal`, `changes`, `verify` on (none of them changes a permission; `verify` only runs when a project check is
  detected, and its command asks for approval like any other command), `step_extension` off (the web has no "continue?"
  event yet, so a turn still ends at the step limit with an `error`). The terminal has all of them on.
- `tool_call` and `tool_result` carry more, additively: `tool_call` also has `risk` and `summary`; `tool_result` also has
  `result` (the full text, capped at 20 000 characters, with `truncated` and `chars`), `ok`, `seconds` and `risk`. The
  web still sends `result_preview` (≤ 200 chars), and a tool that is not found or has bad arguments is still silent.
- `plan`, `verify`, `edit`, `changes`: payloads as in §2.2. The `done` event of the web stays `{"type":"done","reply":…}`.
- A `verify` command that needs an approval suspends the turn like any tool (`awaiting_approval` for `run_command`);
  after the decision the resumed stream starts with the `verify` result (`passed` / `failed` / `skipped`).
- Approvals are unchanged on the web: every CAUTION/DANGEROUS call asks, folder access grants (`/auto-…`) are a
  terminal-only shortcut, and text copied from a web page always asks (`awaiting_approval.from_web: true`, additive).

Reconnect behaviour: if the connection drops mid-turn the UI shows "reconnecting…", then reloads `GET /api/sessions/{id}/messages` and resumes; terminal events are never lost because the final reply is persisted.

## 3. REST endpoints

### 3.1 Existing (EXISTS)
| method & path | purpose | response |
|---|---|---|
| `POST /api/sessions` | new session | `{"id":7,"title":null,"started_at":null,"message_count":0}` |
| `GET /api/sessions` | list | `[SessionOut]` (id, title, started_at, message_count) |
| `DELETE /api/sessions` · `DELETE /api/sessions/{id}` | delete all / one | `{"ok":true}` |
| `GET /api/sessions/{id}/messages` | history | `[{"role":"user|assistant|tool|file_card|block|evidence","content":"…","created_at":null}]` — assistant rows that are tool calls hold JSON `{"tool":…,"arguments":…}`; render those as tool rows, not text |
| `POST /api/sessions/{id}/clear` | wrap up and start fresh | new `SessionOut` |
| `GET/DELETE /api/sessions/{id}/notes` | scratch notes | `{"notes":"…"|null}` / `{"ok":true}` |
| `POST /api/sessions/{id}/messages` | send (SSE) | stream, §2 |
| `GET /api/approvals/pending?session_id=` | open approvals | `[{"id","session_id","tool_name","arguments","risk_level","description","status","created_at"}]` |
| `POST /api/approvals/{id}/decide` | `{"approve":true}` → continues turn | stream, §2 · 404 unknown · 409 already decided |
| `GET /api/models` | `{"current":"…","available":[…],"loaded":[…]}` | |
| `POST /api/models/select` · `POST /api/models/unload` | switch / free VRAM | `{"ok":true}` / `{"unloaded":[…]}` |
| `GET /api/files/{token}` | media/download (HTTP Range supported) | binary |
| `GET /api/images/{job_id}` | image job status | see `image_job` |
| `GET /api/host/stats` | `{"gpu":{"name","vram_used_mb","vram_total_mb","temp_c","fan_percent","utilization_percent","power_watts","power_limit_watts"}|null,"cpu_temp_c","cpu_load_percent","ram":{"used_gb","total_gb"},"local_ip","uptime"}` — any field may be null | |
| `POST /api/host/suspend` | suspend the PC | `{"suspending":true}` |

### 3.2 Built for the GUI (EXISTS) and still planned (NEW)

**EXISTS** (all need the access token; `401` → Connect screen, `429` after 10 wrong attempts per minute):
| area | endpoints | key fields |
|---|---|---|
| Auth | `GET /api/auth/check` | `{ok:true}`; header `Authorization: Bearer <token>` (the token is printed by `agent web-token`) |
| Settings | `GET /api/settings` · `PUT /api/settings/{key}` body `{value}` · `POST /api/settings/{key}/reset` | per item: `key, category, label, help, kind(string\|long_text\|bool\|int\|percent\|choice\|model\|list), value, default, choices?, range?, restart_required`; `404` unknown key, `422 {detail}` invalid value (show `detail`) |
| Memory facts | `GET /api/memory/facts?kind=&q=` · `POST /api/memory/facts` `{content, kind}` · `DELETE /api/memory/facts/{id}` | fact: `{id, content, kind: personal\|incidental, confidence: verified\|assumed, stale, origin: user\|distilled\|web, created_at, updated_at}`; web-learned facts end with `(web, 2026-10-18, domain)` |
| Skills | `GET /api/skills` · `GET /api/skills/{id}` · `DELETE /api/skills/{id}` | `{id,title,problem_description,solution_steps[],tools_used[],tags[],success_count,failure_count,...}` |
| Audit | `GET /api/audit?kind=tool\|shell&q=&limit=` | `[{id, at, kind, detail, approved: 0\|1, result}]` |
| Status | `GET /api/status` | `{loaded_models[], paused_reason: string\|null, game_detected, gpu_mode, chat_model}` |
| GPU mode | `GET /api/gpu` · `PUT /api/gpu` `{mode: single\|dual, ctx?}` | `{gpus:[{name,vendor,total_mb,used_mb}], options:[{mode,label,usable_gb,kv_type}], table:[{model, estimates:[{ctx,fits,over_mb,limited_by_model,approx}\|null]}], chosen:{mode,ctx}, last_mode}` |
| Folder access | `GET /api/access` · `PUT /api/access` `{path, level: ask\|medium\|auto\|greenlist\|ungreenlist}` | `{greenlist:[path], session:[{path,level}]}`; `422 {detail}` when the folder is refused (system or home folders) |
| Source trust | `GET /api/trust/rules` · `PUT /api/trust/rules` `{domain, action: allow\|deny\|tier\|clear, tier?: T1..T4}` | `{allow:[domain], deny:[domain], tiers:{domain:tier}}`. "Block this domain" = `deny`; there is no up/down vote |
| Verify | `POST /api/sessions/{id}/verify` (SSE) | emits `status` (verifying), several `sources` + `verdict` (one pair per counter-search), then `verify_result`, then `done` (`reply` = plain-text report; not a stored chat message). `error` if there is no answer yet |

| Code agent (read-only) | `GET /api/code/sessions` · `GET /api/code/sessions/{id}` (id prefix ok) · `GET /api/code/sessions/{id}/events?kind=&limit=` | list row `{id, root, title, created, updated, steps, model, state: finished\|interrupted (reason)\|in progress, files}`; detail adds `requests[], num_ctx, messages (count), plan, pending, ledger{files{path:…}}`; events `{total, events:[{ts, session, kind: step\|tool\|mask\|compact…, …}], timing:{prefill/generation/tools/waiting…}}` (events are attributed by folder name and time span) |
| Projects | `GET /api/projects` · `GET /api/projects/notes?key=` · `DELETE /api/projects/notes?key=` | `[{key, name, notes}]`, `[{id, note}]`, `{forgotten:n}` |
| Prompt studio | `POST /api/prompt/improve` `{text, answers?}` · `POST /api/prompt/anonymize` `{text, strength: light\|heavy\|poison\|max}` | `{text}`; slow (a model call), show progress; `422` for empty or >20000 chars |

**NEW** (still planned; shapes indicative):
| area | endpoints | key fields |
|---|---|---|
| Memory extras | `PATCH /api/memory/facts/{id}` · `POST /api/memory/prune` · `POST /api/memory/reembed` (job) | |
| Recall trace | `GET /api/sessions/{id}/messages/{idx}/recall` | `{facts:[…], messages:[{role,date,text,similarity}], web_notes:[…]}` |
| Plan / undo | `GET /api/sessions/{id}/plan` · `POST /api/undo` · `GET /api/undo` | undo list: `[{id, at, files:[{path,added,removed}]}]` |
| Code agent control | start / stop a run from the GUI, live event stream (SSE) | today a run is started in the terminal (`agent code`); the GUI can only view runs (see EXISTS) |
| Status extras | workers and per-server reachability | `workers:[{name,state,reason?}]`, `servers:[{name,url,reachable}]` |
| WhatsApp | `GET /api/wa/chats` · `GET /api/wa/search?q=&contact=` | read-only |
| Board | `GET/POST/PATCH/DELETE /api/board/items` · `POST /api/board/items/{id}/refresh` | `{id, block, x, y, w, h, title, refreshable, updated_at}` |

## 4. Blocks — the visual answer catalogue (NEW)

One envelope for all (see the `block` event): `{id, kind, title, spec, source_ids, provenance}`.
`provenance`: `"sources"` (numbers come from cited sources), `"computed"` (calculated by code in a sandbox), `"model"` (from the model's own knowledge — **must be visibly marked as unverified**).
Every block can fail validation: then the UI receives `{"kind":"error","title":"…","spec":{"reason":"…","raw":"…"}}` — design the error state (shows reason + collapsed raw spec).

```jsonc
// table
{"kind":"table","spec":{"columns":[{"key":"name","label":"Language","type":"text"},
   {"key":"speed","label":"Speed","type":"number","unit":"ms","better":"lower"}],
  "rows":[{"name":"Rust","speed":12},{"name":"Python","speed":410}],
  "sort":{"key":"speed","dir":"asc"},"highlight":"best","caption":"Benchmark, 2026"}}

// chart — type: bar|stacked_bar|line|area|pie|donut|scatter|histogram|heatmap|radar
{"kind":"chart","spec":{"type":"line","x":{"label":"Year","type":"category|number|time"},
  "y":{"label":"Price","unit":"€","min":null,"max":null},
  "series":[{"name":"GPU A","points":[[2023,499],[2024,459],[2025,429]]}],
  "annotations":[{"x":2024,"text":"price cut"}],"stacked":false}}

// diagram — mermaid source; type is a hint for the chrome
{"kind":"diagram","spec":{"syntax":"mermaid","type":"flowchart|sequence|state|er|class|gantt|mindmap|timeline|journey|quadrant",
  "source":"flowchart LR\n  A[Ask] --> B{Web needed?}\n  B -->|yes| C[Search]\n  B -->|no| D[Answer]"}}

// compare — options × weighted criteria
{"kind":"compare","spec":{"options":["Model A","Model B"],
  "criteria":[{"name":"Speed","weight":3},{"name":"Quality","weight":5}],
  "scores":[[8,6],[7,9]],"pros":[["fast"],["accurate"]],"cons":[["weaker"],["slow"]],
  "winner":1,"verdict":"B wins on quality, A on speed"}}

// timeline
{"kind":"timeline","spec":{"events":[{"date":"2026-03-02","title":"Beta","text":"…","source_ids":[2]},
                                   {"date":"2026-09-14","title":"Release","text":"…","source_ids":[1,3]}],"orientation":"vertical|horizontal"}}

// stats — KPI tiles
{"kind":"stats","spec":{"tiles":[{"label":"VRAM","value":"21.4","unit":"GB","delta":"+1.2","delta_good":false,
                                  "spark":[3,4,4,6,5,7]}]}}

// callout / quote
{"kind":"callout","spec":{"level":"info|warning|danger|tip","title":"Heads up","text":"…"}}
{"kind":"quote","spec":{"text":"verbatim passage","source_id":1,"highlight":[12,48]}}

// steps — guide / checklist
{"kind":"steps","spec":{"steps":[{"title":"Install","text":"…","command":"uv tool install -e .","done":false}]}}

// code / diff
{"kind":"code","spec":{"language":"python","filename":"game.py","code":"…","highlight_lines":[3,4]}}
{"kind":"diff","spec":{"files":[{"path":"game.py","hunks":[{"old_start":10,"new_start":10,"lines":[{"t":"ctx|add|del","text":"…"}]}]}]}}

// math (LaTeX)
{"kind":"math","spec":{"latex":"E = mc^2","display":true}}

// map (OPEN: needs online tiles)
{"kind":"map","spec":{"points":[{"lat":48.137,"lon":11.575,"label":"Munich"}],"route":null,"offline_fallback":"list"}}

// image / gallery
{"kind":"gallery","spec":{"images":[{"url":"/api/files/<token>","caption":"…","attribution":"© …","source_id":2}]}}

// html — sandboxed interactive widget (opt-in; runs in <iframe sandbox="allow-scripts"> with no network)
{"kind":"html","spec":{"title":"Loan calculator","html":"<!doctype html>…","height":360}}

// svg — sanitized illustration
{"kind":"svg","spec":{"svg":"<svg viewBox=…>…</svg>","alt":"diagram of a lantern"}}
```

Required states for **every** block: skeleton while the tool call is generated (typically 5–80 s; show elapsed), rendered, error, narrow (phone, 360 px), dark/light, print/export, keyboard focus order, accessible alternative text (`alt` or generated summary). Toolbar actions: copy · export (CSV/PNG/SVG/MD as relevant) · expand · pin to Board · "show spec".
Limits to design for: tables up to 500 rows (virtualize), chart up to 10 series × 500 points, diagram source up to 20 kB, widget HTML up to 100 kB.

## 5. Components (design each: states, variants, desktop + phone)

1. **Message bubble** — user · assistant (streaming caret, final) · error · system note. Markdown: h1–h4, lists, task lists, quotes, inline code, code block (language label, copy), tables (horizontal scroll wrapper), links (open externally), math, footnote-style `[n]`.
2. **Status line** — label + animated dots + elapsed seconds (`thinking… 12s`); variants per `state`; a stalled hint after 20 s.
3. **Tool row** — icon, name, one-line summary, duration, expand for arguments/result; running / done / failed / denied / blocked.
4. **Approval card** — risk badge (SAFE not shown), tool, human description, exact arguments (collapsible), Approve/Deny; DANGEROUS adds a type-`yes` field; countdown-free; works as phone sheet; resolved state ("approved · ran 1.2 s", "denied").
5. **Citation marker `[n]`** — inline superscript chip; hover/tap card: domain · trust chip · date · passage.
6. **Trust chip** — `high / medium / low / avoid / unknown` with icon + word; tooltip lists reasons; also `user-allowed` / `user-blocked` variants.
7. **Source card** — see 01 §5.2; states ok · cached · skipped · failed · timeout · blocked · duplicate · suspicious.
8. **Evidence strip** — verdict icon + sentence + counts + "details" (opens inspector); six verdict variants.
9. **Conflict view** — table of attribute × value × sources × trust, preferred row highlighted.
10. **Claim marker** — ⚠ on unsupported sentence/figure with popover.
11. **Blocks** — §4, one component per kind plus shared frame/toolbar/provenance footer.
12. **File card** — audio (artwork, seek, time) · video (poster, subtitle selector info) · image (lightbox) · download; **image-job card** with progress + ETA.
13. **Plan checklist** — item states open/confirmed with evidence tooltip, progress count.
14. **Verify/stuck chips**, **change summary** (files, +/- bars, Undo), **context meter** (bar + numbers + warning at ≥ 70 % and when compaction ran).
15. **Composer** — textarea autosize, attach, send/stop toggle, `/web` toggle, slash palette, attachment chip, disabled states (offline, awaiting approval).
16. **Session list item**, **model picker row** (name, size, loaded dot, current), **model-load bar**, **host badge** (green/amber/red).
17. **Fact row**, **skill card**, **folder-access row** (level selector with consequences text), **audit row**, **setting row** (every `kind`), **GPU option table** (rows = models, columns = options, cells = ctx estimate / "doesn't fit" / "unknown").
18. **Connect screen**, **wake gate**, **offline/unreachable banner**, **toast**, **confirm dialog** (destructive variant), **empty states** for every list.

## 6. Realistic data to mock with

- Long German answer with a 3-level list, a 6-column table and two code blocks.
- A web turn where **source 2 is a "Top 10 best X 2026" affiliate listicle** (trust: avoid, flags `affiliate`,`clickbait`, no author, undated) that *disagrees* with two high-trust sources (verdict: conflicting → resolved to the newest primary).
- A source marked `suspicious_instructions` (the page contained text trying to instruct the assistant) — shown dimmed with "ignored: tried to give the assistant instructions".
- A turn with `blocked` (paywall), `timeout`, `skipped` (pdf) sources next to two `ok` ones.
- A chart whose `provenance` is `model` (unverified banner) vs `computed`.
- A DANGEROUS approval: `rm -r ~/Documents/old` with a 500-character command, and a CAUTION file edit `edit rps/game.py +3 −1`.
- A 45-minute coding run with 12 file edits, 2 failed checks and one undo.
- Context meter at 12 %, 71 % (compaction notice) and 98 %.
- Host stats with a null GPU, and a PC that is asleep.
