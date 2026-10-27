// WhatsApp (design S-WhatsApp, six tabs) over the live routes in agent/web/routes/whatsapp*.py.
// There is no send control anywhere: sending is blocked by policy. Auto-reply is shown as state; the only changes offered make it
// safer (off / draft, loop off, remove a contact). Model work runs as jobs (one at a time) and big-model actions ask first.
// Blur-on-idle (client side, off by default, kept in localStorage) covers every tab.
import { h, debounce } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, confirmDialog, dialog } from "../ui.js";
import { renderMarkdown } from "../md.js";
import { mountView, B, E, G, S, T, X, A, C } from "../viewkit.js";

const BLUR_KEY = "lantern.wa.blur";
const IDLE_MS = 60_000;
const PAGE = 50;
const getBlur = () => { try { return localStorage.getItem(BLUR_KEY) === "1"; } catch (_) { return false; } };
const putBlur = (on) => { try { localStorage.setItem(BLUR_KEY, on ? "1" : "0"); } catch (_) { /* private mode */ } };

const get = (p) => api.get("/api/wa" + p);
const post = (p, b) => api.post("/api/wa" + p, b);
const errText = (e) => (e && (e.detail || e.message)) || "Failed";

const nf = (n) => (n == null ? "—" : Number(n).toLocaleString());
const dur = (s) => (s == null ? "—" : s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m` : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`);
const dt = (v) => { if (v == null || v === "") return "—"; const d = typeof v === "number" ? new Date(v * 1000) : new Date(v); return isNaN(d) ? String(v) : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); };
function when(ts, full = false) {
  const d = new Date(ts * 1000);
  if (isNaN(d)) return "";
  const t = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  const days = (new Date().setHours(0, 0, 0, 0) - new Date(d).setHours(0, 0, 0, 0)) / 864e5;
  if (days < 1 && !full) return t;
  const day = days < 7 ? d.toLocaleDateString(undefined, { weekday: "short" }) : d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
  return full ? `${day} ${t}` : day;
}
const chatName = (c) => c.name || c.chat || "(unnamed)";
const body = (m) => m.text || `[${m.type || "message"}]`;

const NOT_INSTALLED = [E("WhatsApp plugin not installed", "The WhatsApp plugin lives outside the repo (code only, in the plugin folder). Install it, then come back. Existing synced messages can still be read in Chats.", "agent wa --help")];

// state shared between tabs / reloads of this screen
const st = { handle: null, openChat: null, person: null, persona: null, depth: 0, effort: "medium", draft: "", incoming: "", syncDays: 0, k: 24, n: 4, epochs: 4, explain: false, name: "" };

// ------------------------------------------------------------------------------------------------ jobs

const JOB_ST = { running: "running", done: "done", failed: "failed", stopped: "failed" };
function jobPatch(j) {
  return { sub: j.state === "stopped" ? "Stopped." : j.state === "failed" ? j.error : (j.step || (j.stopping ? "Stopping after the current step…" : "")),
    pct: Math.round(j.pct ?? (j.state === "done" ? 100 : 0)), st: JOB_ST[j.state], el: dur(j.elapsed), eta: j.eta ? dur(j.eta) : "—", log: j.log || [] };
}
function jobSection(j, vk) {
  const sec = { k: "job", title: j.label, ...jobPatch(j) };
  if (j.state === "running") {
    sec.acts = [{ l: j.stopping ? "Stopping…" : "Stop", disabled: j.stopping, onClick: async () => { await post(`/jobs/${j.id}/stop`); vk.reload(); } }];
    sec.pollMs = 2000;
    sec.poll = async () => { const n = await get(`/jobs/${j.id}`); if (n.state !== "running") setTimeout(() => vk.reload(), 0); return jobPatch(n); };
  } else sec.acts = [{ l: "Dismiss", onClick: async () => { await api.del(`/api/wa/jobs/${j.id}`); vk.reload(); } }];
  return sec;
}
function mdBox(text) {
  const box = h("div", { class: "vk-panel s-wa-priv s-wa-md", text });
  renderMarkdown(text).then((el) => box.replaceChildren(el)).catch(() => {});
  return box;
}
/** Running jobs of these kinds, and the latest finished one with its text result. */
async function jobsFor(kinds, vk) {
  const all = (await get("/jobs")).filter((j) => kinds.includes(j.kind));
  const out = all.filter((j) => j.state === "running").map((j) => jobSection(j, vk));
  const last = all.find((j) => j.state !== "running");
  if (last) {
    out.push(jobSection(last, vk));
    if (last.state === "done" && last.has_result) {
      const r = (await get(`/jobs/${last.id}`)).result || {};
      const lines = [];
      if (r.text) lines.push(r.text);
      else if (r.empty) lines.push("Nothing to report.");
      for (const k of ["new_messages", "chats", "contacts", "embedded", "merged", "pairs", "train", "val", "dir", "summarized", "skipped", "done", "failed", "file"]) if (r[k] !== undefined) lines.push(`${k.replace("_", " ")}: ${r[k]}`);
      if (lines.length) out.push({ k: "custom", title: `Result: ${last.label}`, render: () => mdBox(lines.join("\n\n")) });
    }
  }
  return out;
}
const start = (vk, path, bodyObj, msg = "Started") => async () => { await post(path, bodyObj); toast(msg); vk.reload({ silent: false }); };
const sure = async (title, text, ok = "Run", danger = false) => confirmDialog(title, text, { confirm: ok, danger });

/** A dropdown of saved markdown files from one of the plugin's folders, shown rendered. */
function fileViewer(kind, title, names, empty) {
  return {
    k: "custom", title,
    render: () => {
      const view = h("div", { class: "s-wa-priv s-wa-md" });
      const sel = h("select", { class: "input vk-sel", "aria-label": title }, names.map((n) => h("option", { value: n, text: n.replace(/\.md$/, "") })));
      const show = async () => {
        if (!sel.value) return;
        view.textContent = "Loading…";
        try { const f = await get(`/digest/files/${kind}/${encodeURIComponent(sel.value)}`); view.replaceChildren(await renderMarkdown(f.text)); } catch (e) { view.textContent = errText(e); }
      };
      sel.addEventListener("change", show);
      if (!names.length) return h("div", { class: "vk-panel vk-help", text: empty });
      show();
      return h("div", { class: "vk-panel" }, sel, view);
    },
  };
}

// ------------------------------------------------------------------------------------------------ chats tab (own widget)

function chatBrowser() {
  const q = h("input", { class: "input s-wa-in", type: "search", placeholder: "Search by keyword", "aria-label": "Search by keyword", autocomplete: "off" });
  const who = h("input", { class: "input s-wa-in", type: "search", placeholder: "Contact", "aria-label": "Contact", autocomplete: "off" });
  const sel = (label, opts) => h("select", { class: "input s-wa-in", "aria-label": label }, opts.map(([v, l]) => h("option", { value: v, text: l })));
  const kind = sel("Chat type", [["all", "All chats"], ["dm", "People"], ["group", "Groups"]]);
  const from = sel("From", [["", "Anyone"], ["me", "Me"], ["them", "Them"]]);
  const range = sel("Date range", [["0", "Any time"], ["1", "Last 24 hours"], ["7", "Last 7 days"], ["30", "Last 30 days"], ["365", "Last year"]]);
  const mtype = sel("Message type", [["", "Any type"], ["text", "Text"], ["image", "Images"], ["video", "Video"], ["audio", "Audio"], ["document", "Documents"], ["sticker", "Stickers"]]);
  const meaning = h("label", { class: "s-wa-check" }, h("input", { type: "checkbox" }), "By meaning");
  const list = h("div", { class: "s-wa-chats l-list", role: "list" });
  const left = h("div", { class: "s-wa-left" }, h("div", { class: "s-wa-search" }, q, who, h("div", { class: "s-wa-row" }, kind, from), h("div", { class: "s-wa-row" }, range, mtype), meaning), list);
  const view = h("div", { class: "s-wa-view" }, h("div", { class: "s-wa-pick" }, icon("wa", 22), "Pick a chat to read it."));
  const wrap = h("div", { class: "s-wa-body" }, left, view);
  let open = null, oldest = null;
  const state = (k, text) => h("div", { class: "l-state" + (k === "error" ? " v-warn" : ""), role: k === "error" ? "alert" : "status" }, text);
  const row = (onclick, on, label, sub, time) => h("button", { class: "s-wa-chat" + (on ? " on" : ""), type: "button", role: "listitem", onclick },
    h("span", { class: "s-wa-ct" }, h("strong", { class: "s-wa-priv", text: label }), h("span", { class: "s-wa-priv s-wa-sub", text: sub })), h("span", { class: "s-wa-time", text: time }));

  async function loadList() {
    const kw = q.value.trim(), who_ = who.value.trim();
    list.replaceChildren(state("loading", "Loading…"));
    try {
      if (kw && meaning.firstChild.checked) {
        const r = await get(`/search/semantic?q=${encodeURIComponent(kw)}&contact=${encodeURIComponent(who_)}`);
        list.replaceChildren(h("pre", { class: "s-wa-priv s-wa-sem", text: r.text || "No matches." }));
      } else if (kw) {
        const p = new URLSearchParams({ q: kw, contact: who_ });
        if (from.value) p.set("who", from.value);
        if (mtype.value) p.set("type", mtype.value);
        if (range.value !== "0") p.set("since", String(Date.now() / 1000 - Number(range.value) * 86400));
        const hits = await get("/search?" + p);
        list.replaceChildren(...(hits.length ? hits.map((m) => row(() => openChat({ id: m.chat_id, name: m.chat }), false, chatName(m), body(m), when(m.timestamp, true))) : [state("empty", "No messages match.")]));
      } else {
        let chats = await get("/chats?kind=" + kind.value);
        if (who_) chats = chats.filter((c) => chatName(c).toLowerCase().includes(who_.toLowerCase()));
        list.replaceChildren(...(chats.length ? chats.map((c) => row(() => openChat(c), open && open.id === c.id, chatName(c), (c.last_from_me ? "You: " : "") + (c.preview || ""), c.last_message_at ? when(c.last_message_at) : ""))
          : [state("empty", who_ ? "No chat with that name." : "No synced WhatsApp messages yet. Use the Sync tab.")]));
      }
    } catch (e) { list.replaceChildren(state("error", errText(e))); }
  }
  const reload = debounce(loadList, 300);
  for (const el of [q, who]) el.addEventListener("input", reload);
  for (const el of [kind, from, range, mtype, meaning]) el.addEventListener("change", loadList);

  const bubble = (m) => h("div", { class: "s-wa-msg s-wa-priv" + (m.from_me ? " me" : ""), role: "listitem" },
    !m.from_me && m.sender && /@g\.us$/.test(open.id) ? h("span", { class: "s-wa-from", text: m.sender.split("@")[0] }) : null,
    h("span", { class: m.text ? "" : "s-wa-media", text: body(m) }), h("span", { class: "s-wa-when", text: when(m.timestamp, true) }));
  async function openChat(c) {
    open = c; oldest = null;
    wrap.classList.add("s-wa-open");
    const thread = h("div", { class: "s-wa-thread", role: "list", "aria-label": "Messages" });
    const older = h("button", { class: "btn btn-secondary s-wa-older hidden", type: "button" }, "Load older messages");
    older.onclick = () => fetchPage(true);
    view.replaceChildren(h("div", { class: "s-wa-head" },
      h("button", { class: "btn btn-ghost s-wa-back", type: "button", "aria-label": "Back to chats", onclick: () => { open = null; wrap.classList.remove("s-wa-open"); loadList(); } }, icon("chevr", 16)),
      h("strong", { class: "s-wa-priv s-wa-title", text: chatName(c) })), h("div", { class: "s-wa-threadbox" }, older, thread));
    async function fetchPage(prepend) {
      try {
        const rows = await get(`/chats/${encodeURIComponent(c.id)}/messages?limit=${PAGE}` + (prepend && oldest ? `&before=${oldest}` : ""));
        if (open !== c) return;
        if (rows.length) oldest = rows[0].timestamp;
        older.classList.toggle("hidden", rows.length !== PAGE);
        const keep = thread.scrollHeight - thread.scrollTop;
        thread.prepend(...rows.map(bubble));
        thread.scrollTop = prepend ? thread.scrollHeight - keep : thread.scrollHeight;
        if (!prepend && !rows.length) thread.replaceChildren(state("empty", "No messages in this chat."));
      } catch (e) { if (open === c) thread.replaceChildren(state("error", errText(e))); }
    }
    await fetchPage(false);
    loadList();
  }
  loadList();
  if (st.openChat) { const id = st.openChat; st.openChat = null; openChat({ id, name: "" }); }
  return wrap;
}

async function chatsTab(vk, blur) {
  const aliases = await get("/aliases").catch(() => []);
  let alias = "", contact = "";
  return [
    B("info", "Read-only", "Nothing is ever sent from here. Search by keyword (or by meaning), filter by chat type, sender, date and message type."),
    G("Privacy", [{ label: "Blur when idle", help: "Hides names and messages on every WhatsApp tab after a minute without interaction, and when the window is hidden. Off by default.", ctl: "sw", val: blur.get(), def: false, onChange: (v) => blur.set(v) }]),
    { k: "custom", title: "Chats", render: chatBrowser },
    G("Nicknames", [
      { label: "Nickname", help: "For a contact that never got a name synced.", ctl: "tx", val: "", def: "", onChange: (v) => { alias = v; } },
      { label: "Contact", help: "Phone number, id or part of a name.", ctl: "tx", val: "", def: "", onChange: (v) => { contact = v; } }]),
    A([{ l: "Save nickname", primary: true, onClick: async (c) => { if (!alias.trim() || !contact.trim()) throw new Error("Fill in both fields."); await post("/aliases", { alias, contact }); toast("Nickname saved"); c.reload({ silent: false }); } }]),
    T("Saved nicknames", ["Nickname", "Contact id"], aliases.map((a) => [a.alias, "m:" + a.jid])),
  ];
}

// ------------------------------------------------------------------------------------------------ sync tab

async function syncTab(vk) {
  const o = await get("/overview");
  if (!o.installed) return NOT_INSTALLED;
  const b = o.bridge || {};
  const out = [];
  if (!b.reachable) out.push(B("warn", "Bridge not running", "The sync needs the WhatsApp bridge. " + (b.error ? `(${b.error})` : ""), [{ l: "Start bridge", primary: true, onClick: async () => { const r = await post("/bridge/start"); toast(r.note || "Bridge starting"); setTimeout(() => vk.reload({ silent: false }), 2500); } }]));
  else if (!b.paired) out.push(B("warn", "Bridge is running but not paired", "Pairing needs the QR code: run `agent wa bridge` in a terminal and scan it (WhatsApp > Linked devices)."));
  out.push(S([["Last sync", dt(o.last_sync_at), ""], ["Messages", nf(o.counts.messages), `${nf(o.counts.chats)} chats`], ["Embedded", `${nf(o.embed.embedded)} / ${nf(o.embed.embeddable)}`, "searchable by meaning"], ["Bridge", !b.reachable ? "not running" : b.connected ? "connected" : "running", b.paired ? "paired" : "not paired"]]));
  const cov = o.embed.embeddable ? Math.round((100 * o.embed.embedded) / o.embed.embeddable) : 100;
  out.push({ k: "job", title: "Embedding messages for semantic search", sub: `${nf(o.embed.embedded)} of ${nf(o.embed.embeddable)} messages have a vector`, pct: cov, st: o.embed.missing ? "queued" : "done", el: "—", eta: "—", acts: [] });
  out.push(...await jobsFor(["sync", "pull", "reingest", "embed"], vk));
  out.push(G("Backfill", [{ label: "Backfill days before syncing", help: "0 = just fetch what is new. Otherwise ask the bridge for this many days of history first.", ctl: "nm", val: st.syncDays, def: 0, onChange: (v) => { st.syncDays = Math.max(0, Math.min(3650, Number(v) || 0)); } }]));
  out.push(A([
    { l: "Sync now", icon: "refresh", primary: true, onClick: (c) => start(c, "/sync", { days: st.syncDays || null })() },
    { l: "Pull without embedding", icon: "download", onClick: (c) => start(c, "/sync/pull")() },
    { l: "Embed missing", icon: "play", disabled: !o.embed.missing, onClick: (c) => start(c, "/sync/embed")() },
    { l: "Re-ingest all", icon: "refresh", onClick: async (c) => { if (await sure("Re-ingest everything?", "Reads the bridge from the very beginning again. Messages are keyed by id, so nothing is duplicated, but it can take a while.", "Re-ingest")) await start(c, "/sync/reingest", { confirm: true })(); } },
    { l: "Merge duplicate chats", onClick: async () => { const r = await post("/sync/merge-lids"); toast(`Merged ${r.merged} duplicate chat(s) from ${r.pairs} known pair(s)`); } },
    { l: "Stop bridge", icon: "stop", disabled: !b.reachable || !o.bridge_owned, onClick: async (c) => { if (await sure("Stop the bridge?", "Live sync stops until you start it again.", "Stop", true)) { await post("/bridge/stop"); c.reload({ silent: false }); } } },
    { l: "Import from backup file…", disabled: true },
  ]));
  out.push(X("Notes", "Import from a backup file is not available yet: the bridge has no import, use the backfill days instead. Stop bridge only works for a bridge started from this app; stop one started in a terminal or service there."));
  const r = o.refresh;
  out.push(T("Background refresh", ["Last pass", "New messages", "Summaries updated", "Skipped"], r ? [[dt(r.at), String(r.new_messages ?? "—"), String(r.summarized ?? "—"), String(r.skipped ?? "—")]] : [], [], false));
  const jobs = (await get("/jobs")).filter((j) => j.state === "failed" || j.state === "stopped");
  out.push(T("Recent errors", ["Time", "Job", "Error"], jobs.slice(0, 8).map((j) => ["m:" + dt(j.started), j.label, j.state === "stopped" ? "neutral:Stopped by you" : "danger:" + (j.error || "failed")])));
  return out;
}

// ------------------------------------------------------------------------------------------------ profiles tab

const PROFILE_KINDS = ["profile", "deep", "summary", "speech", "vibe", "health", "repair", "redflags", "predict", "summaries", "drain", "reflect"];

async function profilesTab(vk) {
  const o = await get("/overview");
  if (!o.installed) return NOT_INSTALLED;
  const [people, queue] = await Promise.all([get("/people"), get("/queue").catch(() => [])]);
  const idx = Math.max(0, people.findIndex((p) => p.jid === st.person));
  const min = o.settings.min_messages;
  const act = (action, label, extra = {}) => ({ l: label, ...extra.btn, onClick: async (c) => {
    const it = c.item;
    if (extra.confirm && !(await sure(extra.confirm[0], extra.confirm[1], "Run", true))) return;
    if (action === "predict" && !st.draft.trim()) throw new Error("Write the draft message to judge in the field below first.");
    await post(`/people/${encodeURIComponent(it.jid)}/jobs`, { action, depth: st.depth || null, effort: st.effort, text: action === "vibe" ? st.incoming : st.draft, confirm: !!extra.confirm });
    toast("Started"); c.reload({ silent: false });
  } });
  const sec = [B("warn", "Deep analysis runs a big model on this PC", "Deep profiles, red-flag scans and reaction predictions use the large model and can take several minutes. Nothing leaves the PC. They ask before they start.")];
  sec.push({
    k: "ld", title: "People and chats", sel: idx, emptyText: "No synced chats yet. Use the Sync tab.",
    items: people.map((p) => ({ t: p.name || p.jid, sub: `${nf(p.total)} messages`, jid: p.jid, group: p.is_group,
      tag: p.is_group ? "neutral:group" : p.deep ? "ok:deep profile" : p.profile ? "ok:profile" : p.too_few ? "warn:too few" : "neutral:no profile" })),
    onSelect: (it) => { st.person = it.jid; },
    detailFor: async (it) => {
      const d = await get(`/people/${encodeURIComponent(it.jid)}`);
      const rows = [["Messages", `${nf(d.stats.total)} (${nf(d.stats.theirs)} from them)`], ["Span", `${dt(d.stats.first_at)} to ${dt(d.stats.last_at)}`]];
      if (d.summary) rows.push(["Summary", d.summary.summary]);
      if (d.profile) rows.push(["Profile", d.profile.profile], ["Based on", `${nf(d.profile.based_on_count)} messages, ${dt(d.profile.updated_at)}`]);
      if (d.speech) rows.push(["Speech", d.speech.profile]);
      if (d.deep) rows.push(["Deep analysis", d.deep.profile], ["Deep based on", `${nf(d.deep.based_on_count)} messages, effort ${d.deep.effort || "?"}`]);
      if (!d.is_group && d.stats.theirs < min) rows.push(["Note", `Too few messages from them (minimum ${min}).`]);
      if (d.queued) rows.push(["Queue", "In the overnight deep-analysis queue"]);
      const acts = d.is_group
        ? [act("summary", "Summarize chat", { btn: { primary: true } }), { l: "Forget summary", danger: true, onClick: async (c) => { if (await sure("Forget this summary?", "The stored summary is deleted. It can be regenerated.", "Forget", true)) { await api.del(`/api/wa/people/${encodeURIComponent(it.jid)}/summary`); c.reload({ silent: false }); } } }]
        : [act("profile", "Build profile", { btn: { primary: true } }), act("summary", "Summarize chat"), act("speech", "Speech profile"), act("vibe", "Vibe check"), act("health", "Friendship health"), act("repair", "Repair suggestions"),
          act("deep", "Deep analysis…", { confirm: ["Run deep analysis?", "This uses the big model on this PC and takes several minutes. Nothing leaves the PC."], btn: { danger: true } }),
          act("redflags", "Red-flag scan…", { confirm: ["Scan for red flags?", "Big model, slow. Read the result with care: it is a model's guess."], btn: { danger: true } }),
          act("predict", "Predict reaction…", { confirm: ["Predict their reaction?", "Judges the draft below on the big model. The draft is NOT sent anywhere."], btn: { danger: true } }),
          { l: d.queued ? "Remove from queue" : "Queue for overnight", onClick: async (c) => { d.queued ? await api.del(`/api/wa/queue/${encodeURIComponent(it.jid)}`) : await post(`/queue/${encodeURIComponent(it.jid)}`); c.reload({ silent: false }); } },
          { l: "Forget summary", danger: true, onClick: async (c) => { if (await sure("Forget this summary?", "Deleted from the database; it can be regenerated.", "Forget", true)) { await api.del(`/api/wa/people/${encodeURIComponent(it.jid)}/summary`); c.reload({ silent: false }); } } },
          { l: "Forget profiles", danger: true, onClick: async (c) => { if (await sure("Forget the stored profiles?", "Deletes the profile and the deep profile of this person. They can be regenerated.", "Forget", true)) { await api.del(`/api/wa/people/${encodeURIComponent(it.jid)}/profile?deep=1`); c.reload({ silent: false }); } } }];
      return { title: d.name, rows, acts };
    },
  });
  sec.push(G("Options for the actions above", [
    { label: "Depth", help: "Analyse the latest N messages (0 = the default; at most 1000, summaries 3000).", ctl: "nm", val: st.depth, def: 0, onChange: (v) => { st.depth = Math.max(0, Number(v) || 0); } },
    { label: "Deep-analysis effort", help: "Deeper means more passes and more messages.", ctl: "seg:low|medium|high|max", val: st.effort, def: "medium", onChange: (v) => { st.effort = v; } },
    { label: "Draft message to judge", help: "Only for Predict reaction. It is read by the model, never sent.", ctl: "ta", val: st.draft, def: "", onChange: (v) => { st.draft = v; } },
    { label: "Read the vibe as if they had just written", help: "Optional, for Vibe check.", ctl: "tx", val: st.incoming, def: "", onChange: (v) => { st.incoming = v; } }]));
  sec.push(...await jobsFor(PROFILE_KINDS, vk));
  sec.push(A([
    { l: "Summarize all chats…", onClick: async (c) => { if (await sure("Summarize all chats?", "Runs the background model over every eligible chat. This can take a long time.", "Start")) await start(c, "/summaries/run", { confirm: true })(); } },
    { l: "Reflect on how I show up", onClick: (c) => start(c, "/reflect", {})() }]));
  sec.push(T("Overnight deep-analysis queue", ["Contact", ""], queue.map((q) => ({ cells: [q.label, { t: "Remove", btn: true, onClick: async () => { await api.del(`/api/wa/queue/${encodeURIComponent(q.jid)}`); vk.reload({ silent: false }); } }] })),
    [{ l: "Run queue now…", disabled: !queue.length, onClick: async (c) => { if (await sure("Run the queue?", "Runs the big model on every queued chat, one after another.", "Run", true)) await start(c, "/queue/run/drain", { effort: st.effort === "medium" ? "high" : st.effort, force: true, confirm: true })(); } }]));
  const refl = (await get("/digest/overview")).files.reflections;
  sec.push(fileViewer("reflections", "Past self-reflections", refl, "No saved reflections yet."));
  const setNum = (key) => async (v) => { const n = Number(v); if (!Number.isInteger(n)) throw new Error("Whole number please."); await api.put("/api/wa/config", { key, value: n }); };
  sec.push(G("Thresholds", [
    { label: "Minimum messages for a profile", help: "Below this the person is skipped.", ctl: "nm", val: o.settings.min_messages, def: 5, onChange: setNum("min_messages") },
    { label: "Messages read for a profile", help: "Most recent messages used when building a profile.", ctl: "nm", val: o.settings.profile_max_messages, def: 400, onChange: setNum("profile_max_messages") },
    { label: "Summary batch size", help: "Messages per summary step.", ctl: "nm", val: o.settings.summarize_batch, def: 200, onChange: setNum("summarize_batch") }]));
  return sec;
}

// ------------------------------------------------------------------------------------------------ digest tab

async function digestTab(vk) {
  const [o, moments, nud] = await Promise.all([get("/digest/overview"), get("/moments").catch(() => null), get("/nudges").catch(() => null)]);
  if (!o.installed) return NOT_INSTALLED;
  const brief = {};
  const sec = [C("Today", [
    { t: "Daily digest", s: o.digest_cursor ? `last made ${dt(o.digest_cursor)}` : "none yet", tag: o.files.digests.length ? "ok:saved" : "neutral:empty" },
    { t: "Daily note", s: o.note ? o.note.note.split("\n")[0].slice(0, 80) : "not written today", tag: o.note ? "ok:saved" : "neutral:empty" },
    { t: "Moments", s: moments ? `${moments.length} from this date in earlier years` : "unavailable", tag: moments && moments.length ? "info:" + moments.length : "neutral:none" },
    { t: "Nudges", s: nud ? `${nud.nudges.length} due, ${nud.hidden} hidden` : "unavailable", tag: nud && nud.nudges.length ? "warn:due" : "neutral:none" }])];
  sec.push(...await jobsFor(["digest", "day_summary", "weekly_summary"], vk));
  sec.push(A([
    { l: "Generate digest", primary: true, onClick: (c) => start(c, "/digest/jobs", { kind: "digest" })() },
    { l: "Preview digest", onClick: (c) => start(c, "/digest/jobs", { kind: "digest", preview: true })() },
    { l: "Day summary", onClick: (c) => start(c, "/digest/jobs", { kind: "day_summary", force: true })() },
    { l: "Weekly summary", onClick: (c) => start(c, "/digest/jobs", { kind: "weekly_summary", force: true })() }]));
  sec.push(X("About the digest", "Generate moves the \"since\" mark forward (like `agent wa digest`); Preview does not. The result appears above while the server runs; the background worker also saves one digest a day."));
  sec.push(G("Daily brief", o.brief.map((b) => ({ label: b.label, help: b.question, ctl: "tx", val: "", def: "", onChange: (v) => { brief[b.label] = v; } }))));
  sec.push(A([{ l: "Save brief", primary: true, onClick: async () => { await api.put("/api/wa/daily-note", { brief }); toast("Saved. The auto-reply AI uses it today."); } }]));
  if (o.note) sec.push(X("Today's note", o.note.note));
  sec.push(fileViewer("digests", "Saved digests", o.files.digests, "No saved digests yet."));
  sec.push(fileViewer("day-summaries", "Day summaries", o.files["day-summaries"], "No day summaries yet."));
  sec.push(fileViewer("weekly-summaries", "Weekly summaries", o.files["weekly-summaries"], "No weekly summaries yet."));
  sec.push(fileViewer("day-log", "Day log (events as messages arrive)", o.files["day-log"], "No day log yet."));
  if (moments) sec.push({ k: "ld", title: "On this day", emptyText: "No messages from this date in earlier years.", items: moments.map((m) => ({ t: m.label, sub: `${m.years_ago} year(s) ago`, tag: "info:" + m.count })),
    detailFor: async (it, i) => ({ title: it.t, rows: moments[i].lines.map((l) => [l.split(":")[0], l.split(":").slice(1).join(":").trim()]).concat(moments[i].count > moments[i].lines.length ? [["More", `and ${moments[i].count - moments[i].lines.length} more`]] : []), acts: [] }) });
  if (nud) sec.push({ k: "ld", title: "Nudges", emptyText: "No friendship has gone unusually quiet.", items: nud.nudges.map((n) => ({ t: n.label, sub: `silent ${n.silent_days} d, usually every ${n.cadence_days} d`, tag: "warn:" + n.overdue + "x overdue", jid: n.jid })),
    detailFor: async (it) => ({ title: it.t, rows: [["Why", "Quiet for much longer than your usual rhythm and you wrote last or they did."], ["Overdue", String(it.tag).replace(/^\w+:/, "")]],
      acts: [{ l: "Open chat", primary: true, onClick: () => { st.openChat = it.jid; st.handle.setTab("chats"); } },
        { l: "Dismiss", onClick: async (c) => { await post(`/nudges/${encodeURIComponent(it.jid)}/hide`, { hours: 720 }); toast("Hidden for 30 days (in this app only)"); c.reload({ silent: false }); } },
        { l: "Snooze 1 h", onClick: async (c) => { await post(`/nudges/${encodeURIComponent(it.jid)}/hide`, { hours: 1 }); c.reload({ silent: false }); } }] }) });
  sec.push(G("Schedule and delivery", [
    { label: "Generate digest at", help: "Not available yet: the background worker makes one digest a day, 24 h after the previous one. There is no time setting.", ctl: "tx", val: "automatic", def: "automatic", disabled: true },
    { label: "Deliver as", help: "Not available yet: digests are shown here and saved to a file; there are no notifications.", ctl: "seg:In app|Notification", val: "In app", def: "In app", disabled: true },
    { label: "Nudges", help: "Always computed from your chat rhythm; there is no switch. Dismiss and snooze only hide them in this app.", ctl: "sw", val: true, def: true, disabled: true }]));
  return sec;
}

// ------------------------------------------------------------------------------------------------ persona tab

async function personaTab(vk) {
  const o = await get("/persona/overview");
  if (!o.installed) return NOT_INSTALLED;
  const sec = [B("danger", "Nothing is ever sent", "Personas are local simulations. They write to you in this window only; the person is not contacted. Auto-reply is shown for review.")];
  const names = {};
  o.cards.forEach((c) => { names[c.jid] = c.name; });
  const all = [...o.cards.map((c) => ({ v: c.jid, l: `${c.name || c.jid} (persona)` })), ...o.candidates.filter((c) => !names[c.jid]).map((c) => ({ v: c.jid, l: `${c.name || c.jid} (${nf(c.message_count)} messages)` }))];
  if (!st.persona || !all.some((a) => a.v === st.persona)) st.persona = all[0] && all[0].v;
  sec.push(C("Personas", o.cards.map((c) => ({ t: c.name || c.jid, s: `built ${dt(c.built_at)}${c.lora ? " · LoRA adapter" : ""}`, tag: c.eval && c.eval.summary ? "ok:tested" : "ok:ready", onClick: () => { st.persona = c.jid; vk.reload({ silent: false }); } }))
    .concat(o.clones.map((c) => ({ t: c.name + " (clone)", s: `trained ${dt(c.created_at)}`, tag: "ok:trained", onClick: async () => {
      const inp = h("input", { class: "input", value: c.name, "aria-label": "New name" });
      if ((await dialog({ title: "Rename clone", body: inp, actions: [{ id: "no", label: "Cancel" }, { id: "yes", label: "Rename", primary: true }] })) === "yes") { await post(`/persona/clones/${encodeURIComponent(c.name)}/rename`, { new_name: inp.value }); vk.reload({ silent: false }); } } })))));
  if (!all.length) { sec.push(E("No chats to clone yet", "A persona needs a 1:1 chat with enough messages from the other person. Sync first.", "agent wa sync")); }
  else {
    sec.push(G("Persona", [
      { label: "Chat", help: "Choose a persona to inspect, or a chat to build one from.", ctl: "sel", opts: all, val: st.persona, def: st.persona, onChange: (v) => { st.persona = v; vk.reload({ silent: false }); } },
      { label: "Display name", help: "Optional. Default: the chat name.", ctl: "tx", val: st.name, def: "", onChange: (v) => { st.name = v; } }]));
    const card = o.cards.find((c) => c.jid === st.persona);
    if (card) {
      const d = await get(`/persona/${encodeURIComponent(st.persona)}`).catch(() => null);
      sec.push(S([["Train set", nf(card.train), "examples"], ["Validation", nf(card.val), "examples"], ["Built", dt(card.built_at), `${nf(card.n_messages)} messages`], ["LoRA adapter", card.lora ? "present" : "none", ""]]));
      if (d) {
        const f = d.fingerprint;
        sec.push(T("Style fingerprint", ["Measure", "Value"], [["Messages / turns", `${nf(f.n_msgs)} / ${nf(f.n_turns)}`], ["Length (median, 10-90 %)", `${Math.round(f.len_p50)} chars (${Math.round(f.len_p10)}-${Math.round(f.len_p90)})`], ["Messages in a row", `${f.burst_mean} (max ${f.burst_max})`],
          ["Lower-case start", `${Math.round(f.lower_start * 100)} %`], ["Ends with a period", `${Math.round(f.ends_period * 100)} %`], ["Emoji", `${Math.round(f.emoji_share * 100)} %`], ["Laughing", `${Math.round(f.laugh_share * 100)} %`], ["Reply time (25/50/75 %)", `${f.latency_p25} / ${f.latency_p50} / ${f.latency_p75} s`], ["Verified facts", String(d.facts)]]));
        sec.push({ k: "custom", title: "Persona card", render: () => mdBox(d.card || "(empty)") });
        const e = d.eval && d.eval.summary;
        sec.push(T("Evaluation", ["Who", "Style distance", "Similarity", "Fooled the judge"], e ? [["Real person", String(e.real.style_distance), "—", "—"], ["This persona", String(e.engine.style_distance), String(e.engine.embedding_similarity), String(e.engine.fooled_judge_rate)], ["Plain prompt (control)", String(e.plain.style_distance), String(e.plain.embedding_similarity), String(e.plain.fooled_judge_rate)]] : []));
      }
    }
    sec.push(...await jobsFor(["persona_build", "persona_eval", "persona_export", "persona_clone"], vk));
    const j = (action, extra = {}) => async (c) => post(`/persona/${encodeURIComponent(st.persona)}/jobs`, { action, name: st.name, k: st.k, n: st.n, epochs: st.epochs, confirm: true, ...extra }).then(() => { toast("Started"); c.reload({ silent: false }); });
    const ask = (title, text, fn, danger = true) => async (c) => { if (await sure(title, text, "Start", danger)) await fn(c); };
    sec.push(A([
      { l: card ? "Rebuild persona card…" : "Build persona card…", primary: true, onClick: ask("Build the persona card?", "The big model reads the whole chat history on this PC (several minutes). Every fact on the card is backed by a verified quote. Nothing is sent.", j("build")) },
      { l: "Evaluate…", disabled: !card, onClick: ask("Evaluate the persona?", "Replays held-out real exchanges and compares them with the persona and a plain-prompt control. Uses the model for a while.", j("eval")) },
      { l: "Prepare training data…", disabled: !card, onClick: ask("Write the training data?", "Writes your chat with this person as a training file (ChatML jsonl) in the personas folder on this PC. Nothing is trained yet.", j("export")) },
      { l: "Train LoRA…", disabled: !o.env.training, onClick: ask("Train the persona?", "Fine-tunes a local model on your GPU for a long time, using this chat. Training stops any other model work. You can stop it.", j("clone")) }]));
    sec.push(G("Evaluation and training options", [
      { label: "Exchanges to replay", ctl: "nm", val: st.k, def: 24, onChange: (v) => { st.k = Math.max(4, Math.min(100, Number(v) || 24)); } },
      { label: "Candidates per reply", help: "More is better and slower.", ctl: "nm", val: st.n, def: 4, onChange: (v) => { st.n = Math.max(1, Math.min(10, Number(v) || 4)); } },
      { label: "Training epochs", help: "More fits closer and takes longer.", ctl: "nm", val: st.epochs, def: 4, onChange: (v) => { st.epochs = Math.max(1, Math.min(20, Number(v) || 4)); } },
      { label: "Show candidate scores in the chat", ctl: "sw", val: st.explain, def: false, onChange: (v) => { st.explain = v; } }]));
    if (!o.env.training) sec.push(X("Training is not available yet", `The training environment (${o.env.dir || "mood-llm"}) is not installed here, so Train LoRA is off. The loss curve and checkpoints are not reported by the training pipeline either, only its stage and progress.`));
    else sec.push(X("Loss and checkpoints", "The training pipeline reports its stage and progress only, so there is no loss curve or checkpoint list. Evaluation shows the quality instead."));
    const hist = [];
    sec.push({ k: "chat", title: "Try the persona (simulation, nothing is sent)", sendLabel: "Ask the persona", msgs: [], onSend: async (text) => {
      if (!card) throw new Error("Build the persona card first.");
      const jb = await post(`/persona/${encodeURIComponent(st.persona)}/jobs`, { action: "say", text, history: hist.slice(-20), n: st.n, explain: st.explain });
      for (let i = 0; i < 400; i++) {
        await new Promise((r) => setTimeout(r, 1500));
        const r = await get(`/jobs/${jb.id}`);
        if (r.state === "running") continue;
        if (r.state !== "done") throw new Error(r.error || "Stopped.");
        hist.push(["me", text]); (r.result.messages || []).forEach((m) => hist.push(["them", m]));
        const sc = (r.result.candidates || []).map((c) => `${c.chosen ? ">" : " "} ${c.src} style ${c.style} sem ${c.sem}${c.ai && c.ai.length ? " AI-isms: " + c.ai.join(", ") : ""}`).join("\n");
        return (r.result.text || "(no usable candidate, try again)") + (sc ? "\n\n" + sc : "");
      }
      throw new Error("The model took too long.");
    } });
    sec.push(X("Chatting with a trained clone", "Talking to a LoRA clone runs in a terminal: `agent wa clone-start NAME`. The chat above uses the persona card engine."));
  }
  const a = await get("/autoreply");
  sec.push(...autoSections(a, vk));
  return sec;
}

function autoSections(a, vk) {
  const out = [S([["Mode", a.mode, a.mode === "auto" ? "can send" : "never sends"], ["Sent today", String(a.sent_today), a.daily_cap ? `cap ${a.daily_cap}/day` : "no cap"], ["Pending drafts", String(a.pending), "held or scheduled"], ["Loop", a.loop ? "on" : "off", a.speed], ["Allowed contacts", String(a.contacts), ""]], "Auto-reply (state)")];
  if (a.mode === "auto") out.push(B("danger", "Auto mode is on outside this app", "Replies can go out to the allowlisted contacts. Switch to draft below to stop all sending; it can only be switched back on with `agent wa auto on` in a terminal."));
  out.push(G("Auto-reply controls (safer direction only)", [
    { label: "Mode", help: "Off or draft (draft holds replies for review and never sends). Turning auto on is not possible from this app.", ctl: "seg:off|draft", val: a.mode === "auto" ? "" : a.mode, def: "off", onChange: async (v) => { await api.put("/api/wa/autoreply/mode", { mode: v }); vk.reload({ silent: false }); } },
    { label: "Loop mode (text first after a silence)", help: a.loop ? "On. Turn it off here; it can only be turned on in a terminal." : "Off. It cannot be turned on from this app.", ctl: "sw", val: a.loop, def: false, disabled: !a.loop, onChange: async (v) => { if (v) throw new Error("Not available from this app."); await api.put("/api/wa/autoreply/loop", { enabled: false }); vk.reload({ silent: false }); } }]));
  out.push(T("Auto-reply per contact", ["Contact", "State", "Drafts", ""], a.allowlist.map((c) => ({ cells: [c.name, c.in_bridge_allowlist ? "info:allowed" : "warn:not in bridge list", String(c.pending),
    c.jid ? { t: "Remove", btn: true, onClick: async () => { if (await sure(`Remove ${c.name}?`, "The auto-responder may no longer text them; queued drafts are cancelled and the bridge allowlist is updated.", "Remove", true)) { await api.del(`/api/wa/autoreply/allow/${encodeURIComponent(c.jid)}`); vk.reload({ silent: false }); } } } : "—"] }))));
  out.push(T("Pending drafts (text is not shown here)", ["Contact", "Status", "Why", "When"], a.pending_rows.map((r) => [r.name, r.status === "held" ? "warn:held" : "info:scheduled", r.reason || "—", "m:" + dt(r.send_at || r.created_at)])));
  out.push(X("Adding contacts", "Allowing a contact, turning auto on, loop on and instant speed are not offered here because they would let the app text people. They exist only as `agent wa auto ...` commands."));
  return out;
}

// ------------------------------------------------------------------------------------------------ safety tab

async function safetyTab(vk) {
  const s = await get("/safety");
  const on = s.send.env_enabled;
  const out = [on ? B("danger", "Sending is enabled outside this app", "AGENT_WA_ALLOW_SEND=1 is set for the server. This app still never offers a send button; sending would need the terminal and asks for approval with the exact message and recipient.")
    : B("block", "Sending is blocked by policy", "This banner stays on every WhatsApp screen. The block is lifted only by an environment variable outside this app.")];
  out.push(G("Policy (read-only)", [{ label: "Send messages", help: on ? "Enabled by the environment." : "Blocked by the product.", ctl: "sw", val: on, def: false, disabled: true }]));
  out.push(S([["Auto-reply mode", s.auto.mode, s.auto.mode === "auto" ? "can send" : "never sends"], ["Allowed contacts", String(s.allowlist), `${s.bridge_allowlist} in the bridge list`], ["Sent today", String(s.auto.sent_today), ""], ["Plugin", s.installed ? "installed" : "missing", ""]]));
  if (s.unsynced.length) out.push(B("warn", "Allowlists differ", `Not in the bridge's own list: ${s.unsynced.join(", ")}. Sends to them would be blocked by the bridge.`));
  out.push(T("Tools the plugin gives the chat agent", ["Tool", "Risk", "What it does"], s.tools.map((t) => ["m:" + t.name, t.risk === "dangerous" ? "danger:dangerous" : "ok:" + t.risk, t.description])));
  out.push(X("If sending is ever enabled elsewhere", "The approval card shows the exact message and recipient and needs you to type \"yes\". This app never offers a send button."));
  return out;
}

// ------------------------------------------------------------------------------------------------ mount

export async function mount(root, ctx) {
  ctx.css("s-whatsapp");
  let blurOn = getBlur(), idle = false, idleTimer = null;
  const host = h("div", { class: "s-wa" });
  const reveal = h("button", { class: "s-wa-reveal hidden", type: "button", onclick: () => wake() }, icon("eyeoff", 14), "Hidden while idle · tap to reveal");
  root.replaceChildren(h("div", { class: "s-wa-root" }, reveal, host));
  const paint = () => { root.firstChild.classList.toggle("s-wa-blurred", blurOn && idle); reveal.classList.toggle("hidden", !(blurOn && idle)); };
  const arm = () => { clearTimeout(idleTimer); if (blurOn) idleTimer = setTimeout(() => { idle = true; paint(); }, IDLE_MS); };
  const wake = () => { idle = false; paint(); arm(); };
  const blur = { get: () => blurOn, set: (on) => { blurOn = on; putBlur(on); idle = false; paint(); arm(); } };
  const onActivity = () => { if (idle) wake(); else arm(); };
  const onHide = () => { if (blurOn && document.hidden) { idle = true; paint(); } };
  for (const ev of ["pointerdown", "keydown", "wheel", "touchstart"]) root.addEventListener(ev, onActivity, { passive: true });
  document.addEventListener("visibilitychange", onHide);
  ctx.onCleanup(() => { clearTimeout(idleTimer); document.removeEventListener("visibilitychange", onHide); });
  arm();
  if (ctx.params && ctx.params.id) st.openChat = decodeURIComponent(ctx.params.id);

  st.handle = mountView(host, ctx, {
    title: "WhatsApp", id: "whatsapp", tab: st.openChat ? "chats" : undefined,
    notice: B("block", "Sending is blocked by policy", "Nothing is ever sent from here. This app has no send button."),
    tabs: [
      { k: "chats", l: "Chats", load: (vk) => chatsTab(vk, blur) },
      { k: "sync", l: "Sync & ingest", load: syncTab, poll: 8000 },
      { k: "profiles", l: "Profiles & analysis", load: profilesTab },
      { k: "digest", l: "Digest & nudges", load: digestTab },
      { k: "persona", l: "Persona & auto-reply", load: personaTab },
      { k: "safety", l: "Safety", load: safetyTab },
    ],
  });
}
