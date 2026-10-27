// Chat screen (design S-Chat): the thread (user bubbles, tool rows, answers with citations, blocks, evidence strip,
// approvals, plan / verify / changes, error states), the composer, and the evidence inspector.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { api, stream, upload } from "../api.js";
import { state, set } from "../store.js";
import { toast, copyText, confirmDialog, dialog } from "../ui.js";
import { mark } from "../mark.js";
import { renderMarkdown } from "../md.js";
import { composer } from "../components/composer.js";
import { renderBlock } from "../components/block.js";
import { approvalCard } from "../components/approval.js";
import { thinking } from "../components/thinking.js";
import { fileCard, imageJobCard } from "../components/filecard.js";
import { verdictBox } from "../components/verdict.js";
import { sourcesPanel } from "../components/sources.js";

// tool name -> [icon, while running, when done, detail from the arguments]
const TOOLS = {
  web_search: ["globe", "searching the web", "searched the web", (a) => a.query], web_fetch: ["file", "reading a page", "read a page", (a) => a.url],
  read_file: ["file", "reading", "read", (a) => a.path], list_dir: ["folder", "listing", "listed", (a) => a.path], find_files: ["search", "searching files", "searched files", (a) => a.pattern || a.name],
  search_files: ["search", "searching files", "searched files", (a) => a.query || a.pattern], write_file: ["pencil", "writing", "wrote", (a) => a.path], replace_in_file: ["pencil", "editing", "edited", (a) => a.path],
  run_command: ["terminal", "running", "ran", (a) => a.command], present: ["chart", "preparing a visual", "showed a visual", (a) => a.title || a.kind], calc: ["cpu", "calculating", "calculated", (a) => a.code],
  remember_this: ["memory", "remembering", "remembered", (a) => a.content], take_note: ["pencil", "taking a note", "took a note", (a) => a.note], send_file: ["file", "sending a file", "sent a file", (a) => a.path],
  generate_image: ["image", "starting an image", "started an image", (a) => a.prompt], current_time: ["clock", "checking the time", "checked the time", () => ""],
};
const toolInfo = (name) => TOOLS[name] || ["terminal", `running ${name}`, `ran ${name}`, () => ""];

export async function mount(root, ctx) {
  let sessionId = ctx.params.id ? Number(ctx.params.id) : null;
  let busy = false, abort = null, lastUser = "", turn = null, nearBottom = true;
  const stats = { sources: null, verdict: null };       // of the latest answer, for the inspector

  set({ ctx: null });
  ctx.onCleanup(() => { if (abort) abort.abort(); set({ ctx: null }); });

  const col = h("div", { class: "l-col" });
  const scroller = h("div", { class: "l-chat-scroll" }, col);
  scroller.addEventListener("scroll", () => { nearBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 140; });
  const scrollDown = (force = false) => { if (force || nearBottom) scroller.scrollTop = scroller.scrollHeight; };

  const comp = composer({
    sessionId, onSend: (text, opts) => send(text, opts), onStop: () => stop(), onCommand: (text) => command(text),
  });
  root.replaceChildren(h("div", { class: "l-chat" }, scroller, h("div", { class: "l-composer-wrap" }, h("div", {}, comp))));
  const applyConnection = () => {
    if (busy) return;
    if (state.connection === "offline") comp.setMode("offline", "Offline — sending is disabled.", "wifioff");
    else if (!turn || !turn.awaiting) comp.setMode("normal");
  };
  const unsub = (await import("../store.js")).subscribe((s, patch) => { if ("connection" in patch) applyConnection(); });
  ctx.onCleanup(unsub);

  // ---------------------------------------------------------------- thread building blocks
  const userBubble = (text) => h("div", { class: "l-user", text });
  const emptyState = () => h("div", { class: "l-empty" }, mark(44, false), h("h2", { text: "Ask anything." }),
    h("p", { text: "This runs on your PC. Nothing leaves it unless a question needs the web, and then you see every source." }),
    h("p", { class: "small" }, h("span", { class: "mono", text: "/" }), " opens commands · ", h("span", { class: "mono", text: "/web" }), " forces a web search for one message · files can be attached."));
  const chipBox = (variant, iconName, headline, text, ...buttons) => h("div", { class: `l-chipbox v-${variant}`, role: "alert" },
    h("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, icon(iconName, 18), h("strong", { text: headline })),
    h("span", { class: "t", text }), buttons.length ? h("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap" } }, ...buttons) : null);

  function openEvidence(view = "sources", activeN = null) {
    if (!stats.sources) return toast("No sources were read for this answer.");
    ctx.shell.openInspector(sourcesPanel({
      sources: stats.sources, verdict: stats.verdict, view, activeN, mode: ctx.shell.isPhone() ? "sheet" : "panel",
      onClose: () => ctx.shell.closeInspector(), onVerify: () => { ctx.shell.closeInspector(); verify(); },
      onFeedback: (action, item) => feedback(action, item), onRules: () => { ctx.navigate("#/settings"); },
    }));
  }
  async function feedback(action, item) {
    const domain = (item.domain || "").replace(/^www\./, "");
    try {
      if (action === "allow") await api.put("/api/trust/rules", { domain, action: "allow" });
      else if (action === "deny") await api.put("/api/trust/rules", { domain, action: "deny" });
      else await api.put("/api/trust/rules", { domain, action: "tier", tier: "T3" });
      toast(action === "deny" ? `${domain} is blocked from now on` : action === "allow" ? `${domain} is trusted from now on` : `${domain} rated lower`);
    } catch (e) { toast(e.detail || "Could not save the rule"); }
  }

  function toolRow(name, args) {
    const [ic, running, done, detail] = toolInfo(name);
    const t0 = Date.now(), time = h("span", { class: "r mono", text: "" });
    const dots = h("span", { class: "dots", "aria-hidden": "true" }, h("i"), h("i"), h("i"));
    const label = h("span", { text: running });
    const d = detail(args || {});
    const code = d ? h("code", { text: String(d).length > 90 ? String(d).slice(0, 89) + "…" : String(d) }) : null;
    const el = h("div", { class: "l-toolrow", dataset: { tool: name } }, dots, icon(ic, 15), label, code, time);
    const row = {
      el, label, code, extras: [],
      finish(failed = false, text = "") {
        const secs = ((Date.now() - t0) / 1000).toFixed(1) + " s";
        el.replaceChildren(icon(failed ? "x" : "check", 15), icon(ic, 15), h("span", { text: text || done }), code, h("span", { class: "r mono", text: failed ? "failed" : secs }), ...row.extras);
        el.classList.toggle("fail", failed);
      },
    };
    return row;
  }

  // ---------------------------------------------------------------- one assistant turn
  function newTurn() {
    const el = h("div", { class: "l-assistant", "aria-busy": "true" });
    col.append(el);
    const t = {
      el, text: "", seg: null, rows: [], row: null, startedAt: Date.now(), waitEl: null, waitTimer: null, awaiting: false, verdict: null, sources: null, blocks: 0, answer: "", hadWeb: false,
    };
    wait(t, "thinking");
    return t;
  }
  function wait(t, state_, detail = "", mood = "") {
    if (!t.waitEl) {
      const th = thinking({ kind: state_, mood });
      const more = h("div", { class: "l-wait hidden" }, icon("clock", 16), h("span", { style: { fontSize: "14px" } }, "still working… ", h("span", { class: "mono s" })),
        h("span", { class: "more", text: "This can take a while. A cold, large prompt can need up to 2½ minutes before the first word." }));
      t.waitEl = h("div", { class: "l-waitwrap" }, th, more);
      t.think = th;
      t.waitT0 = Date.now();
      t.waitTimer = setInterval(() => {
        if (!t.waitEl.isConnected) return clearInterval(t.waitTimer);
        const s = Math.round((Date.now() - t.waitT0) / 1000);
        more.querySelector(".s").textContent = s + " s";
        more.classList.toggle("hidden", s < 20);
      }, 1000);
      t.el.append(t.waitEl);
      scrollDown();
    }
    t.think.set(state_, detail, mood);
  }
  function unwait(t) { if (t.waitEl) { clearInterval(t.waitTimer); if (t.think) t.think.stop(); t.waitEl.remove(); t.waitEl = null; } }
  const place = (t, node) => { if (t.waitEl) t.waitEl.before(node); else t.el.append(node); scrollDown(); };

  async function closeSegment(t) {
    if (!t.seg) return;
    const seg = t.seg; t.seg = null;
    const text = t.text; t.text = "";
    if (!text.trim()) return seg.remove();
    const claims = t.verdict && t.verdict.claims, stray = t.verdict && t.verdict.stray_figures;
    const rendered = await renderMarkdown(text, { onCite: (n) => openEvidence("sources", n), claims, stray });
    seg.replaceWith(rendered);
  }

  function addPlan(t, ev) {
    const items = ev.items || [];
    const box = t.planEl || (t.planEl = h("div", { class: "l-card" }));
    box.replaceChildren(h("div", { class: "l-plan-head" }, h("strong", { text: "Acceptance plan" }), h("span", { class: "mono", text: `${ev.confirmed ?? items.filter((i) => i.status === "confirmed").length} of ${ev.total ?? items.length} confirmed` })),
      h("div", { class: "l-progress ok thin" }, h("i", { style: { width: Math.round((100 * (ev.confirmed || 0)) / Math.max(ev.total || items.length, 1)) + "%" } })),
      ...items.map((i) => h("div", { class: "l-plan-item" + (i.status === "confirmed" ? " done" : "") }, h("span", { style: { marginTop: "2px" } }, icon(i.status === "confirmed" ? "verified" : "clock", 16)),
        h("span", { style: { flex: 1 } }, i.text, i.evidence ? h("span", { class: "ev", text: i.evidence }) : null))));
    if (!box.isConnected) place(t, box);
  }
  function addChip(t, variant, iconName, ...kids) { place(t, h("span", { class: `l-chip v-${variant}`, style: { alignSelf: "flex-start" } }, icon(iconName, 14), ...kids)); }
  function addChanges(t, ev) {
    const files = ev.files || [];
    place(t, h("div", { class: "l-card l-changes" }, h("div", { style: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" } },
      h("strong", { style: { fontSize: "14px", flex: 1 } }, `${files.length} file${files.length === 1 ? "" : "s"} changed `, h("span", { class: "mono a", text: `+${ev.added}` }), " ", h("span", { class: "mono d", text: `−${ev.removed}` })),
      ev.undo_available ? h("button", { class: "btn btn-secondary", style: { minHeight: "36px" }, onclick: () => toast("Undo isn’t available in the web chat yet") }, icon("undo", 15), "Undo") : null),
      ...files.map((f) => h("div", { class: "row" }, h("span", { class: "p", text: f.path }), h("span", { class: "a", text: `+${f.added}` }), h("span", { class: "d", text: `−${f.removed}` }),
        h("span", { class: "bar", "aria-hidden": "true" }, h("i", { style: { flex: f.added || 1, background: "var(--ok)" } }), h("i", { style: { flex: f.removed ? Math.max(f.removed * 4, 6) : 0, background: "var(--danger)" } }))))));
  }

  function handlers(t) {
    return {
      status: (e) => { if (!t.awaiting) wait(t, e.state, e.detail, e.mood); },
      thinking: () => {},
      token: async (e) => {
        unwait(t);
        if (!t.seg) { t.seg = h("div", { class: "l-prose" }); place(t, t.seg); }
        t.text += e.text;
        t.seg.replaceChildren(document.createTextNode(t.text), h("span", { class: "l-caret" }));
        scrollDown();
      },
      tool_call: async (e) => {
        await closeSegment(t);
        const row = toolRow(e.name, e.arguments);
        t.rows.push(row); t.row = row; row.name = e.name;
        if (e.name.startsWith("web_")) t.hadWeb = true;
        place(t, row.el);
        wait(t, "thinking");
      },
      tool_result: (e) => {
        const row = t.rows.slice().reverse().find((r) => r.name === e.name && !r.done) || t.row;
        if (!row) return;
        row.done = true;
        if ((e.result_preview || "").startsWith("BLOCKED by safety policy")) {
          row.el.remove();
          place(t, approvalCard({ tool_name: e.name, risk_level: "BLOCKED", description: e.result_preview, arguments: {} }));
        } else row.finish(/^(Error|Could not|web search failed)/.test(e.result_preview || ""), row.sourcesText || "");
      },
      sources: (e) => {
        t.sources = e; stats.sources = e;
        const row = t.rows.slice().reverse().find((r) => r.name && r.name.startsWith("web_"));
        if (row) {
          const usable = (e.items || []).filter((i) => ["ok", "cached"].includes(i.status) && i.trust && i.trust.level !== "avoid").length;
          row.sourcesText = `read ${(e.items || []).filter((i) => ["ok", "cached"].includes(i.status)).length} pages · ${usable} usable`;
          row.extras = [h("button", { class: "l-link", onclick: () => openEvidence() }, "show details")];
          row.finish(false, row.sourcesText);
        }
      },
      verdict: (e) => { t.verdict = e; stats.verdict = e; },
      file_ready: (e) => place(t, fileCard(e)),
      image_job: (e) => place(t, imageJobCard(e)),
      block: async (e) => { await closeSegment(t); unwait(t); const fig = renderBlock(e); place(t, fig); t.blocks++; wait(t, "thinking"); },
      plan: (e) => addPlan(t, e),
      verify: (e) => { if (e.state === "failed") addChip(t, "danger", "x", `verification failed · retry ${e.attempt}/${e.max_attempts} `, h("span", { class: "mono", text: e.command || "" })); else if (e.state === "passed") addChip(t, "ok", "check", "verification passed"); },
      stuck: (e) => addChip(t, "warn", "search", e.kind === "lookup" ? "same error twice · looking it up" : "same call three times · changing approach"),
      changes: (e) => addChanges(t, e),
      context: (e) => set({ ctx: e }),
      log: () => {},
      awaiting_approval: (e) => {
        unwait(t); t.awaiting = true; busyOff(true);
        const card = approvalCard(e, { onDecide: (ok) => decide(t, e, card, ok) });
        place(t, card);
        comp.setMode("approval", "Waiting for your decision — the turn is paused.");
      },
      error: async (e) => {
        await closeSegment(t); unwait(t);
        const down = e.code === "model_unreachable";
        place(t, down
          ? chipBox("danger", "danger", "The model server isn’t reachable", "Ollama didn’t answer on this PC. Your chats are safe; sending is paused.", h("button", { class: "btn btn-secondary", style: { minHeight: "44px" }, onclick: () => retry() }, "Try again"))
          : chipBox("danger", "danger", "The turn stopped", e.message || "Something went wrong.", h("button", { class: "btn btn-secondary", style: { minHeight: "44px" }, onclick: () => retry() }, icon("refresh", 15), "Retry"),
            h("button", { class: "btn btn-ghost", style: { minHeight: "44px" }, onclick: () => comp.setValue(lastUser) }, "Edit message")));
        t.failed = true;
      },
      done: async (e) => { t.answer = e.reply || ""; await finishTurn(t); },
    };
  }

  async function finishTurn(t) {
    unwait(t);
    // text that streamed in the last segment is the answer; if nothing streamed (a JSON-looking reply), show it now
    if (!t.seg && t.answer.trim() && !t.el.querySelector(".l-prose")) { t.seg = h("div", { class: "l-prose" }); t.el.append(t.seg); t.text = t.answer; }
    if (t.seg && t.answer.trim()) t.text = t.answer;
    await closeSegment(t);
    t.el.removeAttribute("aria-busy");
    if (t.verdict) {
      for (const w of (t.verdict.warnings || []).slice(0, 3)) t.el.append(h("p", { class: "l-note" }, icon("warn", 13), `${w[0].toUpperCase() + w.slice(1)}.`));
      t.el.append(verdictBox(t.verdict, { onDetails: () => openEvidence(t.verdict.label === "conflicting" ? "conflict" : "sources") }));
    }
    if (t.answer.trim()) t.el.append(actions(t));
    scrollDown();
  }

  function actions(t) {
    const btn = (ic, label, fn) => h("button", { class: "l-ghost", onclick: fn }, icon(ic, 15), label);
    return h("div", { class: "l-actions" },
      btn("copy", "Copy", () => copyText(t.answer, "Answer copied")),
      btn("refresh", "Regenerate", () => retry()),
      btn("memory", "Memory used", async () => {
        try { const r = await api.get(`/api/sessions/${sessionId}/messages/-1/recall`); showRecall(r); } catch (e) { toast(e.status === 404 ? "The memory trace isn’t available yet" : e.detail || "Could not load"); }
      }),
      t.sources || stats.sources ? btn("shield-check", "Verify this answer", () => verify()) : null);
  }
  function showRecall(r) {
    const rows = [...(r.facts || []).map((f) => ["fact", f.content || f]), ...(r.messages || []).map((m) => [`${m.role} ${m.date || ""}`, m.text]), ...(r.web_notes || []).map((n) => ["web", n.text || n])];
    dialog({ title: "Memory used for this answer", body: h("div", { class: "l-list", style: { maxHeight: "50vh", overflow: "auto" } }, rows.length ? rows.map(([k, v]) => h("div", { class: "l-card" }, h("span", { class: "mono", text: k }), h("span", { text: v }))) : h("p", { text: "Nothing was recalled for this answer." })), actions: [{ id: "x", label: "Close", primary: true }] });
  }

  // ---------------------------------------------------------------- sending, approvals, verify
  function busyOff(awaiting = false) { busy = awaiting; if (!awaiting) comp.setMode(state.connection === "offline" ? "offline" : "normal", state.connection === "offline" ? "Offline — sending is disabled." : "", "wifioff"); }

  async function ensureSession() {
    if (sessionId) return sessionId;
    const s = await api.post("/api/sessions");
    sessionId = s.id;
    history.replaceState(null, "", "#/chat/" + sessionId);          // same screen, now with an id: no remount
    ctx.params.id = String(sessionId);
    ctx.shell.loadSessions();
    return sessionId;
  }

  async function send(text, { web = false, files = [] } = {}) {
    if (busy) return;
    lastUser = text;
    col.querySelector(".l-empty")?.remove();
    const bubble = userBubble(text || "(attachment)");
    if (files.length) bubble.append(h("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap", marginTop: "6px" } }, files.map((f) => h("span", { class: "l-tag v-neutral" }, icon("file", 13), f.name))));
    col.append(bubble);
    scrollDown(true);
    busy = true; comp.setMode("streaming");
    stats.sources = null; stats.verdict = null;
    turn = newTurn();
    abort = new AbortController();
    try {
      const id = await ensureSession();
      let sent = text;
      if (files.length) {                                // the file goes to the PC first; the message then tells the assistant where it is
        const saved = [];
        for (const f of files) saved.push(await upload(`/api/uploads/${id}/${encodeURIComponent(f.name)}`, f));
        sent = (text ? text + "\n\n" : "") + "[Attached files - read them with read_file]\n" + saved.map((s) => "- " + s.path).join("\n");
      }
      await stream(`/api/sessions/${id}/messages`, { text: sent, web_forced: web }, handlers(turn), abort.signal);
    } catch (e) {
      await handlers(turn).error({ message: e.detail || e.message || "Connection lost." });
    } finally {
      if (!turn.awaiting) { unwait(turn); turn.el.removeAttribute("aria-busy"); busyOff(); refreshTitle(); }
      abort = null;
    }
  }

  async function decide(t, e, card, approve) {
    const resolved = approvalCard(e, { status: approve ? "approved" : "denied", note: approve ? "approved" : "denied · nothing ran" });
    card.replaceWith(resolved);
    t.awaiting = false; busy = true; comp.setMode("streaming");
    wait(t, "thinking");
    abort = new AbortController();
    try { await stream(`/api/approvals/${e.approval_id}/decide`, { approve }, handlers(t), abort.signal); }
    catch (er) { await handlers(t).error({ message: er.detail || er.message }); }
    finally { if (!t.awaiting) { unwait(t); t.el.removeAttribute("aria-busy"); busyOff(); } abort = null; }
  }

  function stop() { if (abort) { abort.abort(); toast("Stopped"); if (turn) { unwait(turn); turn.el.removeAttribute("aria-busy"); closeSegment(turn); } busyOff(); } }
  function retry() { if (lastUser) send(lastUser); }

  async function verify() {
    if (busy || !sessionId) return;
    busy = true; comp.setMode("streaming");
    const t = newTurn();
    t.hadWeb = true;
    abort = new AbortController();
    const h2 = handlers(t);
    try {
      await stream(`/api/sessions/${sessionId}/verify`, {}, {
        ...h2, token: () => {}, done: (e) => { unwait(t); t.el.removeAttribute("aria-busy"); },
        verify_result: (e) => { unwait(t); place(t, verifyCard(e)); },
      }, abort.signal);
    } catch (er) { await h2.error({ message: er.detail || er.message }); }
    finally { unwait(t); busyOff(); abort = null; }
  }
  function verifyCard(r) {
    const V = { supported: ["ok", "check", "supported"], contradicted: ["danger", "danger", "contradicted"], unclear: ["warn", "info", "unclear"] };
    return h("div", { class: "l-card" }, h("div", { class: "l-plan-head" }, h("strong", { text: "Verification" }), h("span", { class: "mono", text: `${r.searches} searches` })),
      ...(r.claims.length ? r.claims.map((c) => { const [v, ic, w] = V[c.status] || V.unclear; return h("div", { style: { display: "flex", flexDirection: "column", gap: "4px", fontSize: "14px" } },
        h("div", { style: { display: "flex", gap: "8px", alignItems: "flex-start" } }, h("span", { class: `l-tag v-${v}` }, icon(ic, 13), w), h("span", { text: c.text })),
        c.status === "supported" ? h("span", { class: "mono", style: { marginLeft: "4px" }, text: "confirmed by " + c.support[0] })
          : c.status === "contradicted" ? h("span", { class: "mono", style: { marginLeft: "4px", color: "var(--danger)" }, text: `a source says ${c.against[0].source} where the answer says ${c.against[0].claim}` })
            : h("span", { class: "mono", style: { marginLeft: "4px" }, text: "no solid source found either way" })); }) : [h("span", { text: "Nothing to check: the last answer has no concrete figures or dates." })]));
  }

  // ---------------------------------------------------------------- slash commands
  function command(text) {
    const [cmd, ...rest] = text.split(/\s+/); const arg = rest.join(" ");
    const lower = cmd.toLowerCase();
    if (lower === "/verify") { verify(); return true; }
    if (lower === "/model") { ctx.navigate("#/models"); return true; }
    if (lower === "/clear") { (async () => {
      if (!sessionId) return toast("Nothing to clear yet");
      try { const s = await api.post(`/api/sessions/${sessionId}/clear`); await ctx.shell.loadSessions(); ctx.navigate("#/chat/" + s.id); toast("Started a fresh conversation"); } catch (e) { toast(e.detail || "Could not clear"); }
    })(); return true; }
    if (lower === "/notes") { (async () => {
      if (!sessionId) return toast("No notes yet");
      try {
        const n = await api.get(`/api/sessions/${sessionId}/notes`);
        const r = await dialog({ title: "Session notes", body: h("pre", { style: { whiteSpace: "pre-wrap", margin: 0, maxHeight: "50vh", overflow: "auto", fontFamily: "var(--font-mono)", fontSize: "13px" }, text: n.notes || "No notes taken this session." }), actions: [{ id: "clear", label: "Clear notes" }, { id: "x", label: "Close", primary: true }] });
        if (r === "clear") { await api.del(`/api/sessions/${sessionId}/notes`); toast("Notes cleared"); }
      } catch (e) { toast(e.detail || "Could not load notes"); }
    })(); return true; }
    if (["/plan", "/project", "/undo"].includes(lower)) { toast(`${cmd} isn’t available in the web chat yet`); return true; }
    const m = /^\/(auto|medium|ask|greenlist|ungreenlist)-(.+)$/i.exec(text.trim());
    if (m) { (async () => {
      try { const r = await api.put("/api/access", { path: m[2].trim(), level: m[1].toLowerCase() }); toast(r.message || "Access updated", 4200); }
      catch (e) { toast(e.detail || "Could not change access", 4200); }
    })(); return true; }
    return false;
  }

  async function refreshTitle() {
    await ctx.shell.loadSessions();
    const s = ctx.shell.sessions.find((x) => x.id === sessionId);
    if (s && s.title) ctx.shell.setTitle(s.title);
  }

  // ---------------------------------------------------------------- history
  async function renderHistory(id) {
    col.append(h("div", { class: "l-skel", style: { height: "14px", width: "70%" } }), h("div", { class: "l-skel", style: { height: "14px", width: "90%" } }), h("div", { class: "l-skel", style: { height: "14px", width: "55%" } }));
    let msgs;
    try { msgs = await api.get(`/api/sessions/${id}/messages`); } catch (e) { col.replaceChildren(chipBox("danger", "danger", "Could not load this chat", e.detail || "")); return; }
    col.replaceChildren();
    if (!msgs.length) { col.append(emptyState()); return; }
    let pending = { sources: null, verdict: null }, rows = [];
    for (const m of msgs) {
      if (m.role === "user") { col.append(userBubble(m.content)); lastUser = m.content; rows = []; pending = { sources: null, verdict: null }; continue; }
      if (m.role === "tool") continue;
      let parsed = null;
      if (m.role === "assistant") { try { const j = JSON.parse(m.content); if (j && j.tool) parsed = j; } catch (_) { /* a real reply */ } }
      if (parsed) { const r = toolRow(parsed.tool, parsed.arguments); r.name = parsed.tool; r.finish(false); r.el.querySelector(".r")?.remove(); col.append(r.el); rows.push(r); continue; }
      if (m.role === "file_card") { try { col.append(fileCard(JSON.parse(m.content))); } catch (_) { /* skip */ } continue; }
      if (m.role === "block") { try { col.append(renderBlock(JSON.parse(m.content))); } catch (_) { /* skip */ } continue; }
      if (m.role === "evidence") {
        try {
          const ev = JSON.parse(m.content);
          if (ev.type === "sources") { pending.sources = ev; stats.sources = ev; const r = rows.slice().reverse().find((x) => x.name && x.name.startsWith("web_")); if (r) { const n = (ev.items || []).filter((i) => ["ok", "cached"].includes(i.status)).length; r.extras = [h("button", { class: "l-link", onclick: () => openEvidence() }, "show details")]; r.finish(false, `read ${n} pages`); r.el.querySelector(".r")?.remove(); } }
          else if (ev.type === "verdict") { pending.verdict = ev; stats.verdict = ev; }
          else if (ev.type === "verify_result") col.append(verifyCard(ev));
        } catch (_) { /* skip */ }
        continue;
      }
      if (m.role === "assistant") {
        const wrap = h("div", { class: "l-assistant" });
        const vd = pending.verdict, src = pending.sources;            // captured now: `pending` is reset below, handlers run later
        const open = (view, n) => { stats.sources = src || stats.sources; stats.verdict = vd || stats.verdict; openEvidence(view, n); };
        wrap.append(await renderMarkdown(m.content, { onCite: (n) => open("sources", n), claims: vd && vd.claims, stray: vd && vd.stray_figures }));
        if (vd) wrap.append(verdictBox(vd, { onDetails: () => open(vd.label === "conflicting" ? "conflict" : "sources") }));
        const answer = m.content, snap = { sources: pending.sources };
        wrap.append(actions({ answer, sources: snap.sources }));
        col.append(wrap); pending = { sources: null, verdict: null }; rows = [];
      }
    }
    try {                                              // a turn that is waiting for a decision survives a reload
      const open = await api.get(`/api/approvals/pending?session_id=${id}`);
      for (const p of open) {
        const t = { el: h("div", { class: "l-assistant" }), awaiting: true };
        col.append(t.el);
        const card = approvalCard({ approval_id: p.id, tool_name: p.tool_name, arguments: p.arguments, risk_level: p.risk_level, description: p.description }, { onDecide: (ok) => decide({ ...t, rows: [], text: "", seg: null, el: t.el, verdict: null, sources: null, answer: "", blocks: 0 }, { approval_id: p.id, tool_name: p.tool_name, arguments: p.arguments, risk_level: p.risk_level, description: p.description }, card, ok) });
        t.el.append(card);
        turn = t; busy = true; comp.setMode("approval", "Waiting for your decision — the turn is paused.");
      }
    } catch (_) { /* approvals are a bonus here */ }
    scrollDown(true);
  }

  if (sessionId) {
    const known = ctx.shell.sessions.find((s) => s.id === sessionId);
    if (known && known.title) ctx.shell.setTitle(known.title);
    comp.setMode("loading");                         // nothing can be sent until the history is in (it would be overwritten)
    await renderHistory(sessionId);
    if (!turn || !turn.awaiting) comp.setMode("normal");
  } else col.append(emptyState());
  applyConnection();
  comp.focus();
}
