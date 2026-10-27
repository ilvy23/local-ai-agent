// Code agent (design S-Code), live: start a goal as a background job, follow and answer it, sessions, policy + risk test, evaluation,
// checkpoints, project memory and statistics. Every control calls a route of routes/code_runs.py or routes/code_admin.py; what the product
// cannot do in a browser (the clarifying questions, merging an agent branch) is shown disabled with the reason.
import { h, relTime, fmtDateTime } from "../dom.js";
import { api, getToken } from "../api.js";
import { dialog, confirmDialog, download, toast } from "../ui.js";
import { renderBlock } from "../components/block.js";
import { mountView, G, S, B, T, A, LG, BR, E, D, X, TH, F } from "../viewkit.js";

const enc = encodeURIComponent;
const clock = (s) => (s == null ? "—" : `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`);
const mins = (s) => (s == null ? "—" : s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`);
const kilo = (n) => (n == null ? "—" : n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)} k` : String(n));
const TONE = { ok: "ok", info: "info", warn: "warn", danger: "danger", neutral: "neutral", block: "block" };
let handle = null;
const go = (k) => handle && handle.setTab(k);
const lines = (t) => String(t || "").split("\n").map((x) => x.trim()).filter(Boolean);

// ---------------------------------------------------------------------------------------------------- shared state (survives tab reloads)
const form = {
  folder: "", goal: "", files: "", engine: "agent", mode: "ask", remember: false, effort: null, diagnose_first: null, auto_commit: null, stages: false,
  autopilot: false, keep_model: false, init_git: false, max_steps: null, stage_max_minutes: null, stage_max_steps: null, stage_prompt_cap: null,
  budget_max_actions: null, budget_max_seconds: null, budget_max_tokens: null, model: null, prose_model: null, escalation_enabled: null,
  escalation_model: null, escalation_budget_s: null, timeout_s: null, memory_gb: null, max_repair_attempts: null, protect: null, test: "",
};
const ui = { run: null, tick: 0, polSel: "", ckFolder: "", ckSearch: "", memFolder: "", sSearch: "", sFolder: "all", pruneDays: 30, evalKind: "fix", seeds: 3, trials: 2,
  evalLog: null, cacheReport: null, benchModel: "", statsSince: "30d" };

/** A dialog that wants one typed word; resolves true only when it matches. */
async function askWord(title, text, word) {
  const inp = h("input", { class: "input", autocomplete: "off", "aria-label": `Type ${word}`, style: { fontFamily: "var(--font-mono)", minHeight: "44px" } });
  const body = h("div", {}, h("p", { text }), h("p", {}, "Type ", h("strong", { class: "mono", text: word }), " to continue."), inp);
  setTimeout(() => inp.focus(), 0);
  const r = await dialog({ title, body, actions: [{ id: "no", label: "Cancel" }, { id: "ok", label: "Continue", danger: true }] });
  if (r !== "ok") return false;
  if (inp.value !== word) { toast(`Not the word ${word}`); return false; }
  return true;
}
/** Calls fn(confirm?). When the server asks for a typed word (loosening), asks and retries once. */
async function withWord(fn, why) {
  try { return await fn(undefined); } catch (e) {
    const m = e && e.status === 422 && /Type (?:the word )?'([a-z]+)'/.exec(e.detail || "");
    if (!m) throw e;
    if (!(await askWord("This gives the agent more room", why || e.detail, m[1]))) throw new Error("Cancelled");
    return fn(m[1]);
  }
}
const errMsg = (e) => (e && (e.detail || e.message)) || String(e);

function unifiedToBlock(text, title = "Changes") {
  const files = []; let cur = null, hunk = null;
  for (const ln of String(text || "").split("\n")) {
    if (ln.startsWith("diff --git")) { cur = { path: (/ b\/(.*)$/.exec(ln) || [])[1] || ln, hunks: [] }; files.push(cur); hunk = null; }
    else if (!cur) continue;
    else if (ln.startsWith("@@")) { hunk = { lines: [{ t: "ctx", text: ln }] }; cur.hunks.push(hunk); }
    else if (hunk && !ln.startsWith("---") && !ln.startsWith("+++") && !ln.startsWith("index ")) hunk.lines.push({ t: ln[0] === "+" ? "add" : ln[0] === "-" ? "del" : "ctx", text: ln.slice(1) });
  }
  return { kind: "diff", title, provenance: "computed", spec: { files: files.length ? files : [{ path: "(no changes)", hunks: [] }] } };
}
async function fetchText(path) {
  const res = await fetch(path, { headers: getToken() ? { Authorization: "Bearer " + getToken() } : {} });
  if (!res.ok) throw new Error((await res.text().catch(() => "")) || `HTTP ${res.status}`);
  return res.blob();
}

/** The folder to show when none was typed: the goal form's, the latest run's, the first folder with a policy. */
async function guessFolder() {
  if (form.folder) return form.folder;
  try { const r = await api.get("/api/code/runs"); if (r.length) return r[0].folder; const p = await api.get("/api/code/policy"); if (p.repos.length) return p.repos[0].path; } catch (_) { /* none */ }
  return "";
}

// ---------------------------------------------------------------------------------------------------- goal
function row(label, help, ctl, key, def, extra = {}) {
  return { label, help, ctl, val: form[key] ?? def, def, onChange: (v) => { form[key] = v; if (extra.after) extra.after(v); }, ...extra, after: undefined };
}
const optionList = (installed, cur, none) => [...(none ? [{ v: "", l: none }] : []), ...[...new Set([...installed, ...(cur ? [cur] : [])])].map((m) => ({ v: m, l: m }))];

async function goalTab(vk) {
  const o = await api.get("/api/code/options" + (form.folder ? `?folder=${enc(form.folder)}` : ""));
  const c = o.coding, classic = form.engine === "classic";
  const defaults = { effort: c.effort, diagnose_first: c.diagnose_first, auto_commit: c.auto_commit, max_steps: (o.policy && o.policy.steps_set) ? o.policy.max_steps : null,
    stage_max_minutes: c.stage_max_minutes, stage_max_steps: c.stage_max_steps, stage_prompt_cap: c.stage_prompt_cap,
    budget_max_actions: c.budget.max_actions, budget_max_seconds: c.budget.max_seconds, budget_max_tokens: c.budget.max_tokens,
    prose_model: c.prose_model, escalation_enabled: c.escalation.tier3_enabled, escalation_model: c.escalation.tier3_model,
    escalation_budget_s: c.escalation.total_budget_seconds, timeout_s: c.sandbox.timeout_seconds, memory_gb: Math.round(c.sandbox.memory_mb / 102.4) / 10,
    max_repair_attempts: c.executor.max_repair_attempts, protect: (c.protect || []).join("\n"), model: classic ? c.executor.model : o.agent_model };
  for (const k of Object.keys(defaults)) if (form[k] === null) form[k] = defaults[k];
  if (form.model === null || form._engineModel !== form.engine) { form.model = defaults.model; form._engineModel = form.engine; }
  const reloadOpts = () => vk.reload({ silent: false });
  const risky = form.mode === "auto" || form.mode === "greenlist";
  const MODES = ["Ask", "Medium", "Auto", "Greenlist"];
  const secs = [
    G("Goal", [
      { label: "Folder", help: "Where the agent may work. A git folder is used as the project root; your home directory and system folders are refused.", ctl: "pa", val: form.folder, def: "",
        onChange: (v) => { form.folder = v.trim(); reloadOpts(); } },
      { label: "Engine", help: "Agent: the tool-using coding agent (stages, autopilot, policy). Classic: write the whole program in one go in a sandbox branch, then repair it (agent code --classic).",
        ctl: "seg:Agent|Classic", val: classic ? "Classic" : "Agent", def: "Agent", onChange: (v) => { form.engine = v.toLowerCase(); reloadOpts(); } },
      { label: "Access level", help: "What runs without asking in this folder, for this run. Greenlist = full access in every future session.", ctl: "seg:" + MODES.join("|"),
        val: MODES.find((m) => m.toLowerCase() === form.mode) || "Ask", def: "Ask", disabled: classic,
        onChange: (v) => { form.mode = v.toLowerCase(); reloadOpts(); } },
      { label: "Remember this level for the folder", help: "Saves the level as the folder's policy (what /mode does in a terminal).", ctl: "sw", val: form.remember, def: false, disabled: classic || form.mode === "ask" || form.mode === "greenlist",
        onChange: (v) => { form.remember = v; } },
      { label: "Goal", help: "Plain words. A resumed session continues with this text.", ctl: "ta", val: form.goal, def: "", onChange: (v) => { form.goal = v; } },
      { label: classic ? "Files to fix" : "Spec / focus files", help: classic ? "Comma separated. With files the classic engine fixes them until the checks pass; without, it builds from scratch." : "Comma separated paths inside the folder; the agent is told to focus on them.",
        ctl: "tx", val: form.files, def: "", onChange: (v) => { form.files = v; } },
      { label: "Check command", help: "The command that proves the work (default: detected from the project).", ctl: "tx", val: form.test, def: "", onChange: (v) => { form.test = v; } },
    ]),
    risky ? B("warn", `${form.mode === "auto" ? "Auto" : "Greenlist"} access: blast radius`, "Edits files and runs scoped commands inside the folder without asking. Anything outside always asks or is refused. You will be asked to type the word to confirm." + (form.mode === "greenlist" ? " Greenlist stays on for every future session until you remove it under Policy." : "")) : B("info", classic ? "Classic engine" : "Agent engine",
      classic ? "Works in a throwaway git worktree on its own branch; nothing in your folder changes until you merge the branch yourself." : "Asks you here, in the run view, before anything its folder policy rates as needing approval. A dangerous command needs the typed word yes."),
    G("Behaviour", [
      { label: "Effort", help: "How much the classic builder writes: min = one file + a small test, mid = balanced, max = may split into files with edge-case tests.", ctl: "seg:Min|Mid|Max", val: (form.effort || "mid").replace(/^./, (x) => x.toUpperCase()), def: defaults.effort.replace(/^./, (x) => x.toUpperCase()),
        disabled: !classic, onChange: (v) => { form.effort = v.toLowerCase(); } },
      row("Diagnose first", "Classic: a separate reasoning pass finds the root cause before each repair.", "sw", "diagnose_first", defaults.diagnose_first, { disabled: !classic }),
      row("Staged mode", "Split a big task into stages, each in a fresh small context (--stages).", "sw", "stages", false, { disabled: classic }),
      row("Autopilot", "A supervisor restarts stopped stages with graded recovery instead of stopping (--autopilot). Also switches on by itself for big tasks.", "sw", "autopilot", false, { disabled: classic }),
      row("Auto-commit", "Classic: commit the accepted work to its branch.", "sw", "auto_commit", defaults.auto_commit, { disabled: !classic }),
      row("Make a baseline commit if needed", "Classic needs a git repository with a commit; this creates one in a folder that has none (never in your home directory or a huge folder).", "sw", "init_git", false, { disabled: !classic }),
      row("Keep the model loaded afterwards", "Do not unload the models when the run ends (--keep-model).", "sw", "keep_model", false, { disabled: classic }),
      { label: "Clarifying questions (intake)", help: "Not available yet: the agent asks them on a terminal; a browser run decides on its own and states its assumptions.", ctl: "sw", val: false, def: false, disabled: true },
    ]),
    G("Budget", [
      row("Steps (blank = automatic)", "Agent: leave blank and it sizes its own budget from the goal (up to 60 steps); a number is a hard cap.", "nm", "max_steps", defaults.max_steps, { disabled: classic }),
      row("Max minutes per stage", "Agent, staged runs. 0 = no limit.", "nm", "stage_max_minutes", defaults.stage_max_minutes, { disabled: classic }),
      row("Max steps per stage", "Agent, staged runs. 0 = no limit.", "nm", "stage_max_steps", defaults.stage_max_steps, { disabled: classic }),
      row("Stage prompt cap (tokens)", "A stage continues in a fresh context when its prompt grows past this. 0 = no cap.", "nm", "stage_prompt_cap", defaults.stage_prompt_cap, { disabled: classic }),
      row("Max actions", "Config budget of the autonomous loop (code-eval --agentic). Used when you save these as defaults; a goal run is limited by the rows above.", "nm", "budget_max_actions", defaults.budget_max_actions),
      row("Max seconds", "Same: autonomous-loop budget.", "nm", "budget_max_seconds", defaults.budget_max_seconds),
      row("Max tokens", "Same: autonomous-loop budget.", "nm", "budget_max_tokens", defaults.budget_max_tokens),
    ]),
    G("Models", [
      { label: classic ? "Executor model" : "Agent model", help: classic ? "coding.executor.model (what code-model sets)." : "This run's model; the folder's policy model or the chat model is the default.", ctl: "sel", opts: optionList(o.installed, form.model),
        val: form.model, def: defaults.model, onChange: (v) => { form.model = v; } },
      { label: "Prose model", help: "A second, CPU-only model for graphic or mature passages (coding.prose_model). None = the write_prose tool does not exist.", ctl: "sel", opts: optionList(o.installed, form.prose_model, "none"),
        val: form.prose_model, def: defaults.prose_model, onChange: (v) => { form.prose_model = v; } },
      row("Escalation to a bigger model", "Classic repair loop: hand a stalled repair to the tier-3 model.", "sw", "escalation_enabled", defaults.escalation_enabled, { disabled: !classic }),
      { label: "Escalation (tier-3) model", help: "", ctl: "sel", opts: optionList(o.installed, form.escalation_model), val: form.escalation_model, def: defaults.escalation_model, disabled: !classic, onChange: (v) => { form.escalation_model = v; } },
      row("Escalation total budget (s)", "", "nm", "escalation_budget_s", defaults.escalation_budget_s, { disabled: !classic }),
    ]),
    G("Sandbox & safety", [
      row("Command timeout (s)", "Classic sandbox: how long one command may run.", "nm", "timeout_s", defaults.timeout_s, { disabled: !classic }),
      row("Memory limit (GB)", "Classic sandbox.", "nm", "memory_gb", defaults.memory_gb, { disabled: !classic }),
      row("Max repair attempts", "Classic fix loop (with files).", "nm", "max_repair_attempts", defaults.max_repair_attempts, { disabled: !classic }),
      row("Protect patterns", "One glob per line: files the agent may never touch (added to the built-in list; it cannot remove any).", "ta", "protect", defaults.protect),
    ]),
    A([
      { l: "Start", icon: "play", primary: true, onClick: () => startRun(vk) },
      { l: "Save as defaults", icon: "download", onClick: async () => {
        await api.put("/api/code/defaults", { values: defaultsPayload() });
        toast("Saved as the defaults in config.yaml"); vk.reload();
      } },
    ]),
  ];
  return secs;
}
function defaultsPayload() {
  const v = {};
  const map = { effort: "effort", diagnose_first: "diagnose_first", auto_commit: "auto_commit", stage_max_minutes: "stage_max_minutes", stage_max_steps: "stage_max_steps",
    stage_prompt_cap: "stage_prompt_cap", budget_max_actions: "budget_max_actions", budget_max_seconds: "budget_max_seconds", budget_max_tokens: "budget_max_tokens",
    prose_model: "prose_model", escalation_enabled: "escalation_enabled", escalation_model: "escalation_model", escalation_budget_s: "escalation_budget_s",
    timeout_s: "timeout_s", max_repair_attempts: "max_repair_attempts", keep_model: "keep_model_loaded" };
  for (const [k, d] of Object.entries(map)) if (form[k] !== null && form[k] !== undefined) v[d] = form[k];
  if (form.memory_gb != null) v.memory_mb = Math.round(Number(form.memory_gb) * 1024);
  if (form.protect != null) v.protect = lines(form.protect);
  if (form.engine === "classic" && form.model) v.executor_model = form.model;
  for (const k of ["stage_max_minutes", "stage_max_steps", "stage_prompt_cap", "budget_max_actions", "budget_max_seconds", "budget_max_tokens", "escalation_budget_s", "timeout_s", "max_repair_attempts", "memory_mb"]) if (k in v) v[k] = Number(v[k]);
  return v;
}

async function startRun(vk, extra = {}) {
  if (!form.folder.trim()) throw new Error("Pick the folder first.");
  const classic = form.engine === "classic";
  const num = (x) => (x === null || x === "" || x === undefined ? undefined : Number(x));
  const body = { folder: form.folder.trim(), goal: form.goal, engine: form.engine, files: form.files.split(",").map((x) => x.trim()).filter(Boolean), test: form.test || undefined, ...extra };
  if (!classic) {
    Object.assign(body, { stages: form.stages, autopilot: form.autopilot, keep_model: form.keep_model, max_steps: num(form.max_steps), stage_max_minutes: num(form.stage_max_minutes),
      stage_max_steps: num(form.stage_max_steps), stage_prompt_cap: num(form.stage_prompt_cap), model: form.model || undefined, new: true });
    if (form.mode !== "ask") body.mode = form.mode === "greenlist" ? "auto" : form.mode;
    body.remember = !!form.remember && form.mode !== "greenlist";
  } else {
    Object.assign(body, { effort: form.effort || undefined, diagnose_first: form.diagnose_first, auto_commit: form.auto_commit, init_git: form.init_git, model: form.model || undefined,
      max_repair_attempts: num(form.max_repair_attempts), escalation_enabled: form.escalation_enabled, escalation_model: form.escalation_model || undefined,
      escalation_budget_s: num(form.escalation_budget_s), timeout_s: num(form.timeout_s), memory_mb: form.memory_gb ? Math.round(Number(form.memory_gb) * 1024) : undefined });
  }
  body.prose_model = form.prose_model ?? undefined;
  body.protect = form.protect != null ? lines(form.protect) : undefined;
  for (const k of ["budget_max_actions", "budget_max_seconds", "budget_max_tokens"]) body[k] = num(form[k]);
  if (form.mode === "greenlist" && !classic) {
    if (!(await askWord("Greenlist this folder", `${form.folder} gets full access in every future session.`, "greenlist"))) throw new Error("Cancelled");
    await api.post("/api/code/policy/greenlist", { folder: form.folder.trim(), confirm: "greenlist" });
  }
  const r = await withWord((word) => api.post("/api/code/runs", { ...body, confirm_access: word }), `Run with ${form.mode} access in ${form.folder}.`);
  ui.run = r.id; ui.tick = 0;
  toast("Run started");
  go("run");
}

// ---------------------------------------------------------------------------------------------------- run
const STATE = { running: ["info", "Running"], waiting: ["warn", "Needs your decision"], paused: ["warn", "Paused"], done: ["ok", "Done"], partial: ["warn", "Partly done"],
  needs_human: ["warn", "Needs a human"], failed: ["danger", "Failed"], stopped: ["neutral", "Stopped"] };

async function runTab(tab, vk) {
  let run = null;
  if (!ui.run) { const list = await api.get("/api/code/runs"); const live = list.find((r) => !r.outcome) || list[0]; if (live) ui.run = live.id; }
  if (ui.run) { try { run = await api.get(`/api/code/runs/${ui.run}`); } catch (e) { if (e.status === 404) ui.run = null; else throw e; } }
  if (!run) { tab.poll = 0; return [E("No run yet", "Start a goal on the first tab. A run executes here in the background; you can leave this screen and come back.", ""), A([{ l: "Start a goal", icon: "play", primary: true, onClick: () => go("goal") }])]; }
  const active = !run.outcome;
  tab.poll = active ? 1500 : 0;
  ui.tick++;
  const [tone, word] = STATE[run.state] || ["neutral", run.state];
  const m = run.meters || {};
  const secs = [];
  secs.push(B(tone === "neutral" ? "info" : tone === "ok" ? "ok" : tone === "danger" ? "danger" : tone === "info" ? "info" : "warn",
    `${word} · ${run.engine === "classic" ? "classic" : "agent"}`, `${run.goal.slice(0, 160)} — ${run.folder}`));
  if (run.approval) {
    secs.push({ k: "approval", call: run.approval, onDecide: async (ok) => { await api.post(`/api/code/runs/${run.id}/decide`, { approve: ok, confirm: ok && run.approval.risk_level === "DANGEROUS" ? "yes" : undefined }); vk.reload(); } });
  }
  const acts = [];
  if (active && run.state !== "paused") acts.push({ l: "Pause", icon: "clock", onClick: async () => { await api.post(`/api/code/runs/${run.id}/pause`); vk.reload(); } });
  if (active && run.state === "paused") acts.push({ l: "Resume", icon: "play", primary: true, onClick: async () => { await api.post(`/api/code/runs/${run.id}/resume`); vk.reload(); } });
  if (active) acts.push({ l: "Stop", icon: "stop", danger: true, onClick: async () => { await api.post(`/api/code/runs/${run.id}/stop`); vk.reload(); } });
  if (!active && run.engine === "agent" && run.session_id) {
    acts.push({ l: "Resume this session", icon: "play", primary: true, onClick: () => resumeSession(vk, { id: run.session_id, root: run.folder }) });
    acts.push({ l: "Undo the last prompt's edits", icon: "undo", onClick: async () => {
      if (!(await confirmDialog("Undo edits", "Restores the files the last prompt changed. Files you changed since are left alone.", { confirm: "Undo", danger: true }))) return;
      const r = await api.post(`/api/code/runs/${run.id}/undo`); toast(r.lines.join(" · ").slice(0, 200)); vk.reload();
    } });
  }
  acts.push({ l: "New goal", icon: "refresh", onClick: () => { ui.run = null; go("goal"); } });
  secs.push(A(acts));
  if (active) secs.push(TH(run.state === "waiting" ? "planning" : "coding", run.folder.split("/").pop()));
  const tiles = [["Time", mins(m.elapsed_s), run.ended ? "total" : "so far"]];
  if (m.steps != null) tiles.unshift(["Steps", m.max_steps ? `${m.steps} steps · budget ${m.max_steps} (${m.budget_source || "user"})` + (m.step_ceiling ? ` · ceiling ${m.step_ceiling}` : "") : String(m.steps), m.budget_source === "auto" ? "sized from the goal, one extension if it progresses" : m.max_steps ? "your cap per prompt" : ""]);
  if (m.ctx_limit) tiles.push(["Context", `${kilo(m.ctx_used)}/${kilo(m.ctx_limit)}`, m.tokens_per_s ? `${m.tokens_per_s} tok/s` : ""]);
  if (m.tokens != null) tiles.push(["Tokens streamed", kilo(m.tokens), ""]);
  if (m.model) tiles.push(["Model", m.model, m.mode ? `${m.mode} access` : ""]);
  secs.push(S(tiles));
  if (run.summary) secs.push(X(active ? "Latest" : "Result", run.summary));
  if (run.result && run.result.not_verified && run.result.not_verified.length) secs.push(B("warn", "Not verified", run.result.not_verified.join("; ")));
  if (run.result && run.result.branch) secs.push(B("info", "Work is on a branch", `git merge ${run.result.branch} (in the project folder). Merging is left to you.`));

  // plan (phase 3: plan-mode view, from planview.py; additive, nothing above depends on it)
  if (run.plan && (run.plan.stages.length || run.plan.checklist.length)) {
    if (run.plan.stages.length) {
      const rows = run.plan.stages.map((s) => [String(s.id), s.title, s.status,
        `${Math.round(s.budget_used_s)}s${s.budget_limit_s ? "/" + Math.round(s.budget_limit_s) + "s" : ""} · ${s.budget_used_steps}${s.budget_limit_steps ? "/" + s.budget_limit_steps : ""} steps`]);
      secs.push(T(`Plan (${run.plan.source})`, ["#", "Stage", "Status", "Budget"], rows, []));
    }
    if (run.plan.checklist.length) {
      const rows = run.plan.checklist.map((i) => [String(i.id), i.done ? "✓" : "", i.text]);
      secs.push(T(`Acceptance checklist (${run.plan.checklist_confirmed}/${run.plan.checklist_total})`, ["#", "", "Item"], rows, []));
    }
    secs.push(X("Right now", run.plan.now + (run.plan.dead_ends ? ` · ${run.plan.dead_ends} dead end(s)` : "")));
  }

  // timeline: the session timeline for agent runs, the run's own steps for classic
  let rows = [];
  if (run.engine === "agent" && run.session_id) {
    try {
      const t = await api.get(`/api/code/sessions/${run.session_id}/timeline`);
      rows = t.items.slice(-80).map((i) => [`m:${clock(i.t)}`, `${i.title}${i.detail ? " · " + i.detail : ""}`, `${TONE[i.tone] || "neutral"}:${i.kind}`]);
      if (t.decision) secs.push(B("warn", "Needs your decision", `${t.decision.text} ${t.decision.task || ""}`.trim(), [{ l: "Continue this session", primary: true, onClick: () => resumeSession(vk, { id: run.session_id, root: run.folder }) }]));
    } catch (_) { /* session not saved yet */ }
  } else rows = run.items.slice(-80).map((i) => [`m:${clock(i.t)}`, `${i.title}${i.detail ? " · " + i.detail : ""}`, `${TONE[i.tone] || "neutral"}:${i.kind}`]);
  secs.push(T("Timeline", ["Time", "Stage / step", "Kind"], rows, []));
  if (!rows.length) secs[secs.length - 1].emptyText = "Nothing recorded yet.";
  // changes
  if (!active || ui.tick % 4 === 1) {
    try { const d = await api.get(`/api/code/runs/${run.id}/diff`); if (d.diff.trim()) secs.push({ k: "block", block: unifiedToBlock(d.diff, "Changes in the folder") }); } catch (_) { /* no repository */ }
  }
  if (run.session_id) {
    try {
      const w = await api.get(`/api/code/why-slow?session=${enc(run.session_id)}`);
      if (!w.empty) secs.push(BR("Why slow: per-step timing", w.parts.map((p) => [p.label, p.pct, `${p.seconds} s`])), ...(w.causes.length ? [X("Likely causes", w.causes.join("\n"))] : []));
    } catch (_) { /* no events */ }
  }
  secs.push(LG("Run log", run.log.slice(-400).map((l) => "I: " + l.text)));
  return secs;
}

async function resumeSession(vk, s) {
  const ta = h("textarea", { class: "input", "aria-label": "Message", rows: 3, style: { width: "100%" } }); ta.value = "continue";
  setTimeout(() => ta.focus(), 0);
  const r = await dialog({ title: "Continue this session", body: h("div", {}, h("p", { text: `${s.root}` }), ta), actions: [{ id: "no", label: "Cancel" }, { id: "ok", label: "Continue", primary: true }] });
  if (r !== "ok") return;
  form.folder = s.root;
  const res = await withWord((word) => api.post("/api/code/runs", { folder: s.root, goal: ta.value || "continue", resume: s.id, new: false, confirm_access: word }), "");
  ui.run = res.id; toast("Resumed"); go("run");
}

// ---------------------------------------------------------------------------------------------------- sessions
async function sessionsTab(vk) {
  const all = await api.get("/api/code/sessions");
  const folders = ["all", ...new Set(all.map((s) => s.root))];
  const q = ui.sSearch.toLowerCase();
  const rows = all.filter((s) => (ui.sFolder === "all" || s.root === ui.sFolder) && (!q || `${s.title} ${s.root}`.toLowerCase().includes(q))).sort((a, b) => b.updated - a.updated);
  const tag = (s) => (s.state === "in progress" ? "info:running" : s.state === "finished" ? "ok:finished" : s.state.startsWith("interrupted") ? "warn:" + s.state : "neutral:" + s.state);
  const btnCell = (t, fn) => ({ t, btn: true, onClick: fn });
  return [
    F([{ l: "Search", v: ui.sSearch, i: "search", input: true, placeholder: "goal or folder", onInput: (v) => { ui.sSearch = v; vk.reload(); } },
       { l: "Folder", v: ui.sFolder, i: "folder", options: folders.map((f) => ({ v: f, l: f })), onInput: (v) => { ui.sFolder = v; vk.reload(); } }]),
    { k: "table", title: "Runs", cols: ["Goal", "Folder", "When", "Result", "", "", "", ""], acts: [],
      rows: rows.map((s) => ({ cells: [s.title, "m:" + s.root, relTime(s.updated), tag(s),
        btnCell("Resume", () => resumeSession(vk, s)), btnCell("State", () => showState(s.root)), btnCell("Export", async () => { download(`trajectory-${s.id}.jsonl`, await fetchText(`/api/code/sessions/${s.id}/export`)); }),
        btnCell("Delete", async () => {
          if (!(await confirmDialog("Delete this session", `${s.title} (${s.root}) is removed for good.`, { confirm: "Delete", danger: true }))) return;
          await api.del(`/api/code/sessions/${s.id}`); vk.reload();
        })] })) },
    G("Housekeeping", [{ label: "Prune sessions older than", help: "days. Sessions that are running are never deleted.", ctl: "nm", val: ui.pruneDays, def: 30, onChange: (v) => { ui.pruneDays = Number(v) || 30; } }]),
    A([{ l: "Preview prune", onClick: async () => { const r = await api.post("/api/code/sessions/prune", { days: ui.pruneDays, dry_run: true }); toast(`${r.would_delete.length} session(s) would be deleted`); } },
       { l: "Prune now", danger: true, onClick: async () => {
         const p = await api.post("/api/code/sessions/prune", { days: ui.pruneDays, dry_run: true });
         if (!p.would_delete.length) return toast("Nothing older than that");
         if (!(await confirmDialog("Delete old sessions", `${p.would_delete.length} saved session(s) untouched for more than ${ui.pruneDays} days are deleted.`, { confirm: "Delete", danger: true }))) return;
         const r = await api.post("/api/code/sessions/prune", { days: ui.pruneDays }); toast(`Deleted ${r.deleted.length}`); vk.reload();
       } }]),
    { ...D("Export the training set", "Writes data/trajectories/sft.jsonl and pairs.jsonl in the repository from every successful saved session (secrets and home paths redacted). It does not train anything.", "export", "Export all"),
      onConfirm: async () => { const r = await api.post("/api/code/sessions/export-all", { confirm: "export" }); toast(`${r.sft} trajectories, ${r.pairs} pairs → ${r.out}`); } },
  ];
}
async function showState(root) {
  toast("Measuring the folder (runs its tests)…");
  const r = await api.get(`/api/code/state?folder=${enc(root)}`);
  await dialog({ title: `Verified state · ${root.split("/").pop()}`, body: h("pre", { class: "mono", style: { maxHeight: "60vh", overflow: "auto", whiteSpace: "pre-wrap", fontSize: "12px" }, text: r.text }) });
}

// ---------------------------------------------------------------------------------------------------- policy
const TIER = { SAFE: "ok:SAFE", CAUTION: "warn:CAUTION", DANGEROUS: "danger:DANGEROUS", BLOCKED: "block:BLOCKED" };

async function policyTab(vk) {
  const p = await api.get("/api/code/policy");
  if (!ui.polSel && p.repos.length) ui.polSel = p.repos[0].path;
  if (!ui.polSel && form.folder) ui.polSel = form.folder;
  const cur = p.repos.find((r) => r.path === ui.polSel) || null;
  const put = (changes, why) => withWord((word) => api.put("/api/code/policy", { folder: ui.polSel, changes, confirm: word }), why);
  const secs = [];
  secs.push({ k: "table", title: "Folders with a policy", cols: ["Folder", "Mode", "Commit", "Network", "Installs", "Steps"], acts: [{ l: "Add folder", onClick: async () => {
    const inp = h("input", { class: "input", "aria-label": "Folder", placeholder: "/home/you/project", style: { minHeight: "44px", width: "100%" } });
    const r = await dialog({ title: "Add a folder", body: h("div", {}, h("p", { text: "It starts on ask: everything asks until you loosen it." }), inp), actions: [{ id: "no", label: "Cancel" }, { id: "ok", label: "Add", primary: true }] });
    if (r !== "ok" || !inp.value.trim()) return;
    ui.polSel = inp.value.trim(); await api.put("/api/code/policy", { folder: ui.polSel, changes: { mode: "ask" } }); vk.reload();
  } }],
    rows: p.repos.map((r) => ({ cells: ["m:" + r.path + (r.exists ? "" : " (gone)"), r.mode === "auto" ? "danger:auto" : r.mode === "medium" ? "warn:medium" : "ok:ask", r.commit, r.network ? "warn:on" : "ok:off", r.installs ? "warn:on" : "ok:off", String(r.max_steps)],
      onClick: () => { ui.polSel = r.path; vk.reload(); } })) });
  if (!p.repos.length) secs[0].emptyText = "No folder has its own policy yet; the first run in a folder asks, or add one here.";
  if (cur) {
    const d = p.defaults;
    secs.push(G(`Policy for ${cur.path}`, [
      { label: "Mode", help: "ask: everything asks. medium: edits are free, commands ask. auto: edits and scoped commands run without asking.", ctl: "seg:ask|medium|auto", val: cur.mode, def: d.mode, onChange: (v) => put({ mode: v }, `Mode ${v} in ${cur.path}.`) },
      { label: "Git commits", help: "May the agent commit?", ctl: "seg:never|ask|auto", val: cur.commit, def: d.commit, onChange: (v) => put({ commit: v }, "") },
      { label: "Network", help: "May commands use the network?", ctl: "sw", val: cur.network, def: d.network, onChange: (v) => put({ network: v }, "Commands may use the network.") },
      { label: "Installs", help: "May commands install packages?", ctl: "sw", val: cur.installs, def: d.installs, onChange: (v) => put({ installs: v }, "Commands may install packages.") },
      { label: "Steps per prompt", help: "5 to 500.", ctl: "nm", val: cur.max_steps, def: d.max_steps, onChange: (v) => put({ max_steps: Number(v) }, "") },
      { label: "Thinking", help: "Reasoning on tool steps (auto = only when stuck).", ctl: "seg:auto|on|off", val: cur.think, def: d.think, onChange: (v) => put({ think: v }, "") },
      { label: "Model", help: "Empty = the chat model.", ctl: "tx", val: cur.model || "", def: "", onChange: (v) => put({ model: v }, "") },
      { label: "Context size", help: "Tokens; empty = the default.", ctl: "nm", val: cur.ctx || "", def: "", onChange: (v) => put({ ctx: v === "" ? null : Number(v) }, "") },
      { label: "Check command", help: "The command that proves a change is good.", ctl: "tx", val: cur.test_cmd || "", def: "", onChange: (v) => put({ test_cmd: v }, "") },
    ]));
    const rules = [...cur.allow.map((x) => ({ x, kind: "allow" })), ...cur.deny.map((x) => ({ x, kind: "deny" }))];
    secs.push({ k: "table", title: "Command rules (allow / deny patterns)", cols: ["Rule", "Pattern", "Effect", ""], acts: [{ l: "Add rule", onClick: async () => {
      const inp = h("input", { class: "input", "aria-label": "Pattern", placeholder: "pytest*", style: { minHeight: "44px", width: "100%" } });
      const sel = h("select", { class: "input", "aria-label": "Effect", style: { minHeight: "44px" } }, h("option", { value: "deny", text: "deny: never runs" }), h("option", { value: "allow", text: "allow: never asks" }));
      const r = await dialog({ title: "Add a command rule", body: h("div", {}, h("p", { text: "A command pattern with * wildcards, matched against the whole command." }), inp, sel), actions: [{ id: "no", label: "Cancel" }, { id: "ok", label: "Add", primary: true }] });
      if (r !== "ok" || !inp.value.trim()) return;
      const kind = sel.value; await put({ [kind]: [...cur[kind], inp.value.trim()] }, `Allow ${inp.value.trim()} without asking.`); vk.reload();
    } }],
      rows: rules.map((r, i) => ({ cells: [String(i + 1), "m:" + r.x, r.kind === "allow" ? "ok:allow" : "block:blocked", { t: "Remove", btn: true, onClick: async () => {
        await put({ [r.kind]: cur[r.kind].filter((y) => y !== r.x) }, `Removing a deny rule gives the agent more room.`); vk.reload();
      } }] })), emptyText: "No rules: the built-in safety classifier decides alone." });
    secs.push(A([{ l: "Forget this folder's policy", danger: true, onClick: async () => {
      if (!(await confirmDialog("Forget the policy", `${cur.path} goes back to the defaults (ask).`, { confirm: "Forget", danger: true }))) return;
      await api.del(`/api/code/policy?folder=${enc(cur.path)}`); ui.polSel = ""; vk.reload();
    } }]));
  }
  secs.push({ k: "table", title: "Greenlist (full access in every session)", cols: ["Folder", ""], acts: [{ l: "Greenlist a folder", danger: true, onClick: async () => {
    const inp = h("input", { class: "input", "aria-label": "Folder", placeholder: "/home/you/project", style: { minHeight: "44px", width: "100%" } });
    const r = await dialog({ title: "Greenlist a folder", body: h("div", {}, h("p", { text: "Full access (edits and scoped commands) in every future session." }), inp), actions: [{ id: "no", label: "Cancel" }, { id: "ok", label: "Continue", danger: true }] });
    if (r !== "ok" || !inp.value.trim()) return;
    if (!(await askWord("Greenlist", inp.value.trim(), "greenlist"))) return;
    await api.post("/api/code/policy/greenlist", { folder: inp.value.trim(), confirm: "greenlist" }); vk.reload();
  } }], rows: p.greenlist.map((g) => ({ cells: ["m:" + g, { t: "Remove", btn: true, onClick: async () => { await api.del(`/api/code/policy/greenlist?folder=${enc(g)}`); vk.reload(); } }] })), emptyText: "Empty." });
  secs.push({ k: "test", title: "Test a command", cmd: "", onTest: async (cmd) => {
    const r = await api.post("/api/code/policy/classify", { command: cmd, folder: ui.polSel || undefined });
    const verdict = r.verdict ? `In ${ui.polSel.split("/").pop()} (${r.mode}): ${r.verdict}${r.reason ? " — " + r.reason : ""}.` : "Pick a folder above to see what its policy does.";
    return { tier: TIER[r.tier] || r.tier, why: `${r.why ? r.why + " " : ""}${verdict}` };
  } });
  return secs;
}

// ---------------------------------------------------------------------------------------------------- evaluation
async function evalTab(tab, vk) {
  const [ev, cmp, mdl] = await Promise.all([api.get("/api/code/evals"), api.get("/api/code/model-compare"), api.get("/api/code/models")]);
  const secs = [B("warn", "Evaluations use the GPU", "A run starts the real coding model on held-out tasks and takes a long time. Only one job runs at a time, and not while a goal run is active.")];
  if (ev.active) secs.push({ k: "job", title: ev.active.label, sub: ev.active.model, pct: 0, st: "running", el: mins(Date.now() / 1000 - ev.active.started), eta: "—", log: ev.active.log.slice(-8), pollMs: 2000,
    poll: async () => { const j = await api.get(`/api/code/jobs/${ev.active.id}`); const st = j.state === "running" ? "running" : j.state === "done" ? "done" : "failed"; if (st !== "running") setTimeout(() => vk.reload(), 500); return { st, log: j.log.slice(-8), pct: st === "done" ? 100 : 0 }; },
    acts: [{ l: "Stop", danger: true, onClick: async () => { await api.post(`/api/code/jobs/${ev.active.id}/stop`); vk.reload(); } }] });
  secs.push(G("Run an evaluation", [
    { label: "Loop", help: "Fix loop: pass@k on held-out tests (agent code-eval). Agentic: the open-ended loop on multi-file goals.", ctl: "seg:fix|agentic", val: ui.evalKind, def: "fix", onChange: (v) => { ui.evalKind = v; } },
    { label: "Seeds per task", help: "Fix loop: 3 or more for real numbers.", ctl: "nm", val: ui.seeds, def: 3, onChange: (v) => { ui.seeds = Number(v); } },
    { label: "Trials per task", help: "Agentic loop (slow).", ctl: "nm", val: ui.trials, def: 2, onChange: (v) => { ui.trials = Number(v); } },
  ]), A([{ l: "Run eval", icon: "play", primary: true, disabled: !!ev.active, onClick: async () => {
    if (!(await askWord("Run an evaluation", `Runs ${mdl.coding} on the GPU.`, "run"))) return;
    await api.post("/api/code/evals", { kind: ui.evalKind, seeds: ui.seeds, trials: ui.trials, confirm: "run" }); vk.reload();
  } }]));
  secs.push({ k: "table", title: `code-eval runs (${ev.tasks.length} fix tasks)`, cols: ["Kind", "Model", "Pass rate", "Duration", "When", ""],
    rows: ev.runs.map((r) => ({ cells: [r.kind.replace("eval-", ""), "m:" + (r.model || "—"), r.total ? `${r.passed >= r.total * 0.8 ? "ok" : "warn"}:${r.passed}/${r.total} · ${Math.round(100 * r.passed / r.total)} %` : (r.exit_code === 0 ? "neutral:no score" : "danger:failed"),
      mins(r.seconds), relTime(r.started), { t: "Log", btn: true, onClick: async () => { const l = await api.get(`/api/code/evals/history/${r.id}`); ui.evalLog = { id: r.id, lines: l.log }; vk.reload(); } }] })), emptyText: "No evaluation was recorded here yet." });
  secs.push({ k: "table", title: "Model comparison (from the recorded fix-loop runs)", cols: ["Model", "Pass rate", "Median time", "Runs"],
    rows: cmp.rows.map((r) => ({ cells: ["m:" + r.model, `${r.pass_rate >= 80 ? "ok" : "warn"}:${r.pass_rate} %`, mins(r.median_s), String(r.runs)] })), emptyText: "Run the fix-loop evaluation with different coding models to compare them here." });
  secs.push(G("Coding model (code-model)", [
    { label: "Executor model", help: "What the classic builder and the evaluation use. " + (mdl.routing_override ? "models.coding is set and is kept in sync." : ""), ctl: "sel", opts: optionList(mdl.installed, mdl.coding), val: mdl.coding, def: mdl.coding,
      onChange: async (v) => { await api.put("/api/code/models/coding", { model: v }); toast(`Coding model: ${v}`); } },
  ]));
  secs.push(X("Recommended coders", mdl.recommended.map((r) => `${r.model}: ${r.note}${mdl.installed.includes(r.model) ? " (installed)" : " (not pulled)"}`).join("\n")));
  if (ui.evalLog) secs.push(LG(`Per-task log: ${ui.evalLog.id}`, ui.evalLog.lines.map((l) => (/fail|error|✗/i.test(l) ? "E: " : "I: ") + l)));
  tab.poll = ev.active ? 0 : 0;
  return secs;
}

// ---------------------------------------------------------------------------------------------------- checkpoints
async function checkpointsTab(vk) {
  if (!ui.ckFolder) ui.ckFolder = await guessFolder();
  const secs = [G("Folder", [{ label: "Project folder", help: "Checkpoints are private git refs the agent makes before the first edit of a prompt, plus the agent-code/* branches of classic runs.", ctl: "pa", val: ui.ckFolder, def: "", onChange: (v) => { ui.ckFolder = v.trim(); vk.reload(); } }])];
  if (!ui.ckFolder) return [...secs, E("Pick a folder", "Enter the project folder to list its checkpoints.", "")];
  const data = await api.get(`/api/code/checkpoints?folder=${enc(ui.ckFolder)}`);
  const items = data.items;
  const showDiff = async (it, against) => {
    const d = await api.get(`/api/code/checkpoints/detail?folder=${enc(ui.ckFolder)}&ref=${enc(it.ref)}&against=${enc(against)}`);
    await dialog({ title: `${it.short} vs ${against === "now" ? "now" : against === "previous" ? "previous" : against}`, body: h("div", { style: { maxHeight: "60vh", overflow: "auto" } }, renderBlock(unifiedToBlock(d.diff, `+${d.added} −${d.removed}`))) });
  };
  secs.push({ k: "ld", title: "Git checkpoints", emptyText: "No checkpoints yet: the agent makes one before its first edit of a prompt.",
    items: items.map((i) => ({ t: `${i.short} · ${i.kind === "branch" ? "agent branch" : i.subject || "checkpoint"}`, sub: fmtDateTime(i.when), tag: i.kind === "branch" ? "info:branch" : "", it: i })),
    detailFor: async (item) => {
      const it = item.it;
      const [prev, now] = await Promise.all([api.get(`/api/code/checkpoints/detail?folder=${enc(ui.ckFolder)}&ref=${enc(it.ref)}&against=previous`), api.get(`/api/code/checkpoints/detail?folder=${enc(ui.ckFolder)}&ref=${enc(it.ref)}&against=now`)]);
      return { title: it.ref, rows: [["Files", String(prev.files.length)], ["Diff vs previous", `+${prev.added} −${prev.removed}`], ["Diff vs now", now.files.length ? `+${now.added} −${now.removed} in ${now.files.length} file(s)` : "none"]],
        acts: [
          it.kind === "checkpoint" ? { l: "Restore", primary: true, onClick: async () => {
            if (!(await askWord("Restore this checkpoint", `Files in ${ui.ckFolder} are overwritten with this checkpoint. A safety checkpoint of the current state is made first.`, "restore"))) return;
            const r = await api.post("/api/code/checkpoints/restore", { folder: ui.ckFolder, ref: it.ref, confirm: "restore" });
            toast(r.safety_checkpoint ? `Restored. Undo with ${r.safety_checkpoint}` : "Restored"); vk.reload();
          } } : { l: "Copy merge command", onClick: async () => { await navigator.clipboard.writeText(`git merge ${it.ref}`); toast("Copied: merging is done in your terminal"); } },
          { l: "Diff vs previous", onClick: () => showDiff(it, "previous") }, { l: "Diff vs now", onClick: () => showDiff(it, "now") },
          { l: "Diff between…", onClick: async () => {
            const sel = h("select", { class: "input", "aria-label": "Other checkpoint", style: { minHeight: "44px" } }, items.filter((x) => x.ref !== it.ref).map((x) => h("option", { value: x.ref, text: `${x.short} · ${fmtDateTime(x.when)}` })));
            const r = await dialog({ title: "Diff between two checkpoints", body: sel, actions: [{ id: "no", label: "Cancel" }, { id: "ok", label: "Show", primary: true }] });
            if (r === "ok" && sel.value) await showDiff(it, sel.value);
          } }] };
    } });
  return secs;
}

// ---------------------------------------------------------------------------------------------------- project memory
async function memoryTab(vk) {
  if (!ui.memFolder) ui.memFolder = await guessFolder();
  const secs = [];
  if (ui.memFolder) {
    const m = await api.get(`/api/code/memory?folder=${enc(ui.memFolder)}`);
    secs.push(G(m.folder, [
      { label: "Project folder", ctl: "pa", val: ui.memFolder, def: "", onChange: (v) => { ui.memFolder = v.trim(); vk.reload(); } },
      { label: "Notes", help: "Kept per repository, one per line (8 to 300 characters). The agent adds its own with project_note; newest first.", ctl: "ta", val: m.notes.map((n) => n.note).join("\n"), def: "",
        onChange: async (v) => { const r = await api.put("/api/code/memory/notes", { folder: ui.memFolder, notes: v }); if (r.skipped.length) return { error: `Too short to keep: ${r.skipped.join(" | ")}` }; return { val: r.notes.map((n) => n.note).join("\n") }; } },
      { label: "Check that passes", help: "Command that proves a change is good (the folder policy's test command).", ctl: "tx", val: m.test_cmd, def: "", onChange: async (v) => { await api.put("/api/code/memory/check", { folder: ui.memFolder, test_cmd: v }); } },
    ]));
    secs.push(D("Forget this project", `Removes the notes and the check command for ${m.folder}.`, "forget", "Forget project"));
    secs[secs.length - 1].onConfirm = async () => { const r = await api.del(`/api/code/memory?folder=${enc(m.folder)}`); toast(`Forgot ${r.forgotten} note(s)`); vk.reload(); };
    secs.push({ k: "table", title: "Projects with notes", cols: ["Project"], rows: m.repos.map((r) => ({ cells: ["m:" + r.key], onClick: () => { ui.memFolder = r.key; vk.reload(); } })), emptyText: "No project has notes yet." });
  } else secs.push(G("Project", [{ label: "Project folder", help: "Enter a folder to see its notes.", ctl: "pa", val: "", def: "", onChange: (v) => { ui.memFolder = v.trim(); vk.reload(); } }]));
  return secs;
}

// ---------------------------------------------------------------------------------------------------- stats
async function statsTab(tab, vk) {
  const [st, speed, bench] = await Promise.all([api.get(`/api/code/stats${ui.statsSince === "all" ? "" : "?since=" + ui.statsSince}`), api.get("/api/code/stats/speed"), api.get("/api/code/bench")]);
  const o = st.overall;
  const secs = [F([{ l: "Period", v: ui.statsSince, i: "clock", options: ["7d", "30d", "all"].map((x) => ({ v: x, l: x === "all" ? "all time" : "last " + x })), onInput: (v) => { ui.statsSince = v; vk.reload(); } }])];
  secs.push(S([["Runs", String(st.runs), ui.statsSince === "all" ? "all time" : "last " + ui.statsSince], ["Finished", st.runs ? `${Math.round(100 * st.finished / st.runs)} %` : "—", `${st.finished} of ${st.runs}`],
    ["Prefix-cache hit", "—", "not measured by the engine (see the cache report)"], ["Speed", st.median_tps ? `${st.median_tps} tok/s` : "—", "median generation"]]));
  if (o) {
    secs.push(BR("Where the time goes (all runs)", [["Prefill", o.prefill_pct, `${o.prefill_pct} %`], ["Generation", o.decode_pct, `${o.decode_pct} %`], ["Tools", o.tool_pct, `${o.tool_pct} %`], ["Compaction", o.compact_pct, `${o.compact_pct} %`], ["Other", o.other_pct, `${o.other_pct} %`]]));
    secs.push(S([["Steps", String(o.steps), `${o.steps_per_min}/min`], ["Reads : edits : runs", `${o.reads}:${o.edits}:${o.runs}`, ""], ["Loops", String(o.loops), `${o.loop_aborts} aborts`], ["Stage caps", String(o.stage_cap), `${o.stage_resplit} re-splits`]], "Behaviour"));
  }
  secs.push({ k: "table", title: "Recent sessions", cols: ["Session", "Outcome", "Wall", "Steps", "Prefill"], rows: st.sessions.slice(-12).reverse().map((r) => ({ cells: ["m:" + (r.title || r.session), (r.outcome || "").startsWith("finished") || (r.outcome || "").startsWith("autopilot done") ? "ok:" + r.outcome : "neutral:" + String(r.outcome).slice(0, 40), mins(r.wall_s), String(r.steps), `${r.share_pct.prefill} %`] })), emptyText: "No recorded sessions in this period." });
  secs.push(speed.empty ? X("Speed trial", "No event log yet.") : { k: "table", title: `Speed trial · latest session (${speed.session})`, cols: ["Check"], rows: speed.lines.map((l) => [(/^PASS/.test(l) ? "ok:" : /^FAIL/.test(l) ? "danger:" : "neutral:") + l]) });
  secs.push(A([{ l: "Prefix-cache report", icon: "refresh", onClick: async () => { toast("Building a sample context…"); ui.cacheReport = await api.post("/api/code/stats/cache-report"); vk.reload(); } }]));
  if (ui.cacheReport) {
    secs.push(BR("Fixed prompt prefix (tokens, largest parts)", ui.cacheReport.parts.slice(0, 8).map((p, i, a) => [p.name, Math.round(100 * p.tokens / (a[0].tokens || 1)), `${p.tokens}`])));
    secs.push(X(ui.cacheReport.breakers.length ? "Prefix-cache breakers" : "Prefix-cache breakers: none found", [...ui.cacheReport.breakers, ...ui.cacheReport.notes].join("\n")));
  }
  if (bench.active) secs.push({ k: "job", title: bench.active.label, sub: "", pct: 0, st: "running", el: mins(Date.now() / 1000 - bench.active.started), eta: "—", log: bench.active.log.slice(-8), pollMs: 2500,
    poll: async () => { const j = await api.get(`/api/code/jobs/${bench.active.id}`); const stt = j.state === "running" ? "running" : j.state === "done" ? "done" : "failed"; if (stt !== "running") setTimeout(() => vk.reload(), 500); return { st: stt, log: j.log.slice(-8), pct: stt === "done" ? 100 : 0 }; },
    acts: [{ l: "Stop", danger: true, onClick: async () => { await api.post(`/api/code/jobs/${bench.active.id}/stop`); vk.reload(); } }] });
  secs.push({ k: "table", title: "Server benchmark", cols: ["Run", "Result", "When", ""], acts: [
    { l: "Show plan (sends nothing)", disabled: !!bench.active, onClick: async () => { await api.post("/api/code/bench", { kind: "dry" }); vk.reload(); } },
    { l: "Run benchmark", danger: true, disabled: !!bench.active, onClick: async () => {
      if (!(await askWord("Run the server benchmark", "Sends real generation requests and keeps the GPUs busy for a long time.", "benchmark"))) return;
      await api.post("/api/code/bench", { kind: "server", confirm: "benchmark" }); vk.reload();
    } }],
    rows: bench.runs.map((r) => ({ cells: [r.label, r.exit_code === 0 ? "ok:finished" : r.stopped ? "neutral:stopped" : "danger:failed", relTime(r.started), { t: "Output", btn: true, onClick: async () => { const l = await api.get(`/api/code/evals/history/${r.id}`); await dialog({ title: r.label, body: h("pre", { class: "mono", style: { maxHeight: "60vh", overflow: "auto", whiteSpace: "pre-wrap", fontSize: "12px" }, text: l.log.join("\n") }) }); } }] })), emptyText: "No benchmark was run from here yet." });
  tab.poll = bench.active ? 0 : 0;
  return secs;
}

// ---------------------------------------------------------------------------------------------------- mount
export async function mount(root, ctx) {
  const tabs = [
    { k: "goal", l: "Start a goal", load: goalTab },
    { k: "run", l: "Run view", poll: 0, load: (vk) => runTab(tabs[1], vk) },
    { k: "sessions", l: "Sessions", load: sessionsTab },
    { k: "policy", l: "Policy editor", load: policyTab },
    { k: "eval", l: "Evaluation", load: (vk) => evalTab(tabs[4], vk) },
    { k: "checkpoints", l: "Checkpoints", load: checkpointsTab },
    { k: "memory", l: "Project memory", load: memoryTab },
    { k: "stats", l: "Stats", load: (vk) => statsTab(tabs[7], vk) },
  ];
  // open on the run view when something is working right now
  let first = "goal";
  try { const runs = await api.get("/api/code/runs"); const live = runs.find((r) => !r.outcome); if (live) { ui.run = live.id; first = "run"; } } catch (_) { /* the tabs report their own errors */ }
  handle = mountView(root, ctx, { title: "Code agent", tab: first, tabs });
}
