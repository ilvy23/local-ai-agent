// Memory (design S-Memory): facts, per-answer trace, session notes, web cache and the danger zone.
import { h, debounce } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, confirmDialog, dialog } from "../ui.js";

const TABS = [["facts", "Facts"], ["trace", "Per-answer trace"], ["notes", "Session notes"], ["cache", "Web cache"], ["danger", "Danger zone"]];
const fmtDate = (v) => { const d = new Date(typeof v === "number" ? v * 1000 : v); return isNaN(d) ? "" : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }); };
const span = (s) => { const a = Math.abs(s); return a >= 86400 ? `${Math.round(a / 86400)} d` : a >= 3600 ? `${Math.round(a / 3600)} h` : `${Math.max(1, Math.round(a / 60))} min`; };
const loading = (text) => h("div", { class: "l-state", role: "status" }, h("div", { class: "l-skeleton", style: { width: "100%" } }), text);
const failed = (what, e, retry) => h("div", { class: "l-state", role: "alert" }, h("strong", { text: `${what} could not be loaded.` }), h("span", { class: "mono", text: e.detail || "Is the server running?" }),
  h("button", { class: "btn btn-secondary", onclick: retry }, icon("refresh", 15), "Try again"));
const tag = (variant, iconName, text, dashed = false) => h("span", { class: `l-tag v-${variant}` + (dashed ? " s-memory-dashed" : "") }, icon(iconName, 13), text);

export async function mount(root, ctx) {
  ctx.css("s-memory");
  ctx.shell.setTitle("Memory");
  let tab = TABS.some(([id]) => id === ctx.params.id) ? ctx.params.id : "facts";
  let alive = true;
  ctx.onCleanup(() => { alive = false; });

  const body = h("div", { class: "s-memory-body" });
  const tabs = h("div", { class: "s-memory-tabs", role: "tablist", "aria-label": "Memory" });
  root.replaceChildren(h("div", { class: "s-memory" }, tabs, body));

  function show(id) {
    tab = id;
    tabs.replaceChildren(...TABS.map(([tid, label]) => h("button", { class: "s-memory-tab", role: "tab", id: `s-memory-tab-${tid}`, "aria-selected": String(tid === id), "aria-controls": "s-memory-panel", tabindex: tid === id ? "0" : "-1",
      onclick: () => show(tid),
      onkeydown: (e) => { const i = TABS.findIndex(([x]) => x === id), n = { ArrowRight: 1, ArrowLeft: -1 }[e.key]; if (n) { show(TABS[(i + n + TABS.length) % TABS.length][0]); tabs.querySelector('[aria-selected="true"]').focus(); } } }, label)));
    body.setAttribute("role", "tabpanel"); body.id = "s-memory-panel"; body.setAttribute("aria-labelledby", `s-memory-tab-${id}`);
    body.replaceChildren();
    ({ facts, trace, notes, cache, danger })[id]();
  }

  // ---------------------------------------------------------------- facts
  function facts() {
    let kind = "", q = "";
    const list = h("div", { class: "l-list" });
    const search = h("input", { class: "input s-memory-search", type: "search", placeholder: "Search memory — meaning or exact words", "aria-label": "Search facts" });
    const seg = h("div", { class: "seg", role: "radiogroup", "aria-label": "Kind" }, [["", "All"], ["personal", "Personal"], ["incidental", "Incidental"]].map(([v, l]) =>
      h("label", { class: "seg-opt" }, h("input", { type: "radio", name: "s-memory-kind", checked: v === "", onchange: () => { kind = v; refresh(); } }), l)));
    body.append(h("div", { class: "l-toolbar" }, h("div", { class: "s-memory-searchwrap" }, search, h("span", { class: "s-memory-searchicon" }, icon("search", 16))), seg,
      h("button", { class: "btn btn-primary", onclick: add }, icon("plus", 15), "Add fact"), h("button", { class: "btn btn-secondary", onclick: prune }, "Prune…")), list);
    search.addEventListener("input", debounce(() => { q = search.value; refresh(); }, 250));

    async function refresh() {
      list.replaceChildren(loading("Loading facts…"));
      let rows;
      try { rows = await api.get(`/api/memory/facts?kind=${kind}&q=${encodeURIComponent(q)}`); } catch (e) { return list.replaceChildren(failed("Facts", e, refresh)); }
      if (!alive || tab !== "facts") return;
      if (!rows.length) {
        list.replaceChildren(q || kind ? h("div", { class: "l-state" }, "No facts match.") : h("div", { class: "s-memory-empty" }, h("h2", { text: "No memories yet." }),
          h("p", { text: "Facts you state, and things learned in passing, will be listed here. You can edit or forget any of them." }),
          h("div", {}, h("button", { class: "btn btn-primary", onclick: add }, icon("plus", 15), "Add a fact"))));
        return;
      }
      list.replaceChildren(...rows.map(row));
    }
    function row(f) {
      const personal = f.kind === "personal";
      const conf = f.confidence === "verified" ? tag("ok", "verified", "verified") : f.stale ? tag("danger", "clock", "assumed · re-check (older than 24 h)") : tag("warn", "single", "assumed");
      const origin = { user: "stated by you", web: "learned from web", distilled: "distilled from chat" }[f.origin] || f.origin;
      return h("div", { class: "s-memory-row" },
        h("div", { class: "s-memory-line" }, h("span", { class: "s-memory-text", text: f.content }),
          h("button", { class: "s-memory-ib", "aria-label": "Edit fact", title: "Edit", onclick: () => edit(f) }, icon("pencil", 16)),
          h("button", { class: "s-memory-ib", "aria-label": "Forget fact", title: "Forget", onclick: () => forget(f) }, icon("trash", 16))),
        h("div", { class: "s-memory-meta" }, personal ? tag("info", "user", "personal · always remembered") : tag("neutral", "memory", "incidental · learned in passing", true), conf,
          h("span", { class: "s-memory-dim", text: `${origin} · updated ${fmtDate(f.updated_at)}` })));
    }
    function factForm(initial) {
      const text = h("textarea", { class: "input s-memory-area", rows: "3", maxlength: "1500", "aria-label": "Fact", placeholder: "e.g. Prefers answers in German unless asked otherwise." });
      text.value = initial.content;
      const radios = ["personal", "incidental"].map((v) => h("label", { class: "seg-opt" }, h("input", { type: "radio", name: "s-memory-fact-kind", value: v, checked: v === initial.kind }), v));
      return { el: h("div", { class: "s-memory-form" }, text, h("div", { class: "seg", role: "radiogroup", "aria-label": "Kind" }, radios),
        h("span", { class: "s-memory-dim", text: "Personal facts go into every answer. Incidental ones are only recalled when relevant." })),
        read: () => ({ content: text.value.trim(), kind: radios.find((l) => l.querySelector("input").checked).querySelector("input").value }) };
    }
    async function ask(title, initial, actionLabel) {
      const form = factForm(initial);
      const r = await dialog({ title, body: form.el, actions: [{ id: "no", label: "Cancel" }, { id: "yes", label: actionLabel, primary: true }] });
      const v = form.read();
      return r === "yes" && v.content ? v : null;
    }
    async function add() {
      const v = await ask("Add a fact", { content: "", kind: "personal" }, "Add");
      if (!v) return;
      try { await api.post("/api/memory/facts", v); toast("Fact added"); show("facts"); } catch (e) { toast(e.detail || "Could not add the fact"); }
    }
    async function edit(f) {
      const v = await ask("Edit fact", f, "Save");
      if (!v) return;
      try { await api.patch(`/api/memory/facts/${f.id}`, v); toast("Fact updated"); refresh(); } catch (e) { toast(e.detail || "Could not save the fact"); }
    }
    async function forget(f) {
      if (!(await confirmDialog("Forget this fact?", `“${f.content}” will no longer be recalled.`, { confirm: "Forget", danger: true }))) return;
      try { await api.del(`/api/memory/facts/${f.id}`); refresh(); } catch (e) { toast(e.detail || "Could not forget the fact"); }
    }
    async function prune() {
      let dry;
      try { dry = await api.post("/api/memory/prune", { dry_run: true }); } catch (e) { return toast(e.detail || "Could not check for junk facts"); }
      if (!dry.count) return toast("No junk facts found");
      const text = h("div", { class: "s-memory-form" }, h("p", { text: "Facts that look like file paths, tool output or timestamps, captured by mistake:" }),
        h("ul", { class: "s-memory-junk" }, dry.facts.map((t) => h("li", { text: t }))), dry.count > dry.facts.length ? h("span", { class: "s-memory-dim", text: `…and ${dry.count - dry.facts.length} more` }) : null);
      if ((await dialog({ title: `Forget ${dry.count} junk fact${dry.count === 1 ? "" : "s"}?`, body: text, actions: [{ id: "no", label: "Cancel" }, { id: "yes", label: "Forget them", danger: true }] })) !== "yes") return;
      try { const r = await api.post("/api/memory/prune", {}); toast(`Forgot ${r.count} fact${r.count === 1 ? "" : "s"}`); refresh(); } catch (e) { toast(e.detail || "Could not prune"); }
    }
    refresh();
  }

  // ---------------------------------------------------------------- session picker shared by trace + notes
  async function sessionPicker(onPick) {
    let sessions;
    try { sessions = await api.get("/api/sessions?limit=50"); } catch (e) { body.append(failed("Sessions", e, () => show(tab))); return; }
    if (!alive) return;
    const withMsgs = sessions.filter((s) => s.message_count > 0);
    if (!withMsgs.length) { body.append(h("div", { class: "l-state" }, "No conversations yet.")); return; }
    const sel = h("select", { class: "input s-memory-select", "aria-label": "Conversation" }, withMsgs.map((s) => h("option", { value: s.id, text: `${s.title || "Untitled"} · ${s.message_count} messages` })));
    const current = Number(location.hash.match(/#\/chat\/(\d+)/)?.[1]);
    if (withMsgs.some((s) => s.id === current)) sel.value = String(current);
    sel.addEventListener("change", () => onPick(Number(sel.value)));
    body.append(h("div", { class: "l-toolbar" }, h("label", { class: "s-memory-dim", for: "s-memory-session", text: "Conversation" }), Object.assign(sel, { id: "s-memory-session" })));
    onPick(Number(sel.value));
  }

  // ---------------------------------------------------------------- per-answer trace
  function trace() {
    body.append(h("p", { class: "s-memory-lede", text: "Pick an answer to see which memories were recalled to write it. From a chat, the “Memory used” button shows the same." }));
    const answers = h("div", { class: "l-list" }), result = h("div", { class: "s-memory-trace" });
    sessionPicker(async (sid) => {
      answers.replaceChildren(loading("Loading answers…")); result.replaceChildren();
      let msgs;
      try { msgs = await api.get(`/api/sessions/${sid}/messages`); } catch (e) { return answers.replaceChildren(failed("Messages", e, () => show("trace"))); }
      const items = [];
      msgs.forEach((m, i) => { if (m.role === "assistant") items.push({ i, q: [...msgs.slice(0, i)].reverse().find((x) => x.role === "user")?.content || "", a: m.content }); });
      if (!items.length) return answers.replaceChildren(h("div", { class: "l-state" }, "This conversation has no answers yet."));
      answers.replaceChildren(...items.slice(-12).reverse().map((it) => h("button", { class: "s-memory-answer", "aria-pressed": "false", onclick: (e) => {
        answers.querySelectorAll("[aria-pressed=true]").forEach((b) => b.setAttribute("aria-pressed", "false")); e.currentTarget.setAttribute("aria-pressed", "true"); load(sid, it);
      } }, h("strong", { text: it.q.slice(0, 90) || "(no question)" }), h("span", { class: "s-memory-dim", text: it.a.slice(0, 110) }))));
    }).then(() => body.append(answers, result));

    async function load(sid, it) {
      result.replaceChildren(loading("Loading the trace…"));
      let r;
      try { r = await api.get(`/api/sessions/${sid}/messages/${it.i}/recall`); }
      catch (e) { return result.replaceChildren(h("div", { class: "l-state", role: "alert" }, e.status === 404 ? "No trace is available for this answer." : e.detail || "Could not load the trace.")); }
      const sim = (v) => (v == null ? null : h("span", { class: "s-memory-sim" }, h("span", { class: "mono", text: `match ${v.toFixed(2)}` }), h("span", { class: "s-memory-bar", "aria-hidden": "true" }, h("i", { style: { width: `${Math.max(0, Math.min(1, v)) * 100}%` } }))));
      const sect = (title, rows, empty) => [h("h5", { text: title }), rows.length ? rows : h("div", { class: "s-memory-dim", text: empty })];
      result.replaceChildren(
        h("p", { class: "s-memory-lede", text: `Opened from the reply “${it.q.slice(0, 60)}${it.q.length > 60 ? "…" : ""}”. These were recalled to write it.` }),
        ...sect("Facts", (r.facts || []).map((f) => h("div", { class: "s-memory-row" }, h("span", { text: f.content }), h("div", { class: "s-memory-meta" }, sim(f.similarity), h("span", { class: "s-memory-dim", text: f.kind })))), "No facts were used."),
        ...sect("Past messages", (r.messages || []).map((m) => h("div", { class: "s-memory-row" }, h("span", { class: "s-memory-quote", text: `“${m.text}”` }), h("span", { class: "s-memory-dim", text: `${m.role === "user" ? "you" : m.role} · ${fmtDate(m.date)} · match ${Number(m.similarity).toFixed(2)}` }))), "No earlier messages were recalled."),
        ...sect("Web notes", (r.web_notes || []).map((n) => h("div", { class: "s-memory-row" }, h("span", { text: n.text }), n.date ? h("span", { class: "s-memory-dim", text: `noted ${fmtDate(n.date)}` }) : null)), "No web notes were used."));
    }
  }

  // ---------------------------------------------------------------- session notes
  function notes() {
    body.append(h("p", { class: "s-memory-lede", text: "The scratchpad the assistant keeps for one conversation. It is not part of long-term memory." }));
    const out = h("div", { class: "s-memory-trace" });
    sessionPicker(async (sid) => {
      out.replaceChildren(loading("Loading notes…"));
      let r;
      try { r = await api.get(`/api/sessions/${sid}/notes`); } catch (e) { return out.replaceChildren(failed("Notes", e, () => show("notes"))); }
      if (!r.notes) return out.replaceChildren(h("div", { class: "l-state" }, "No notes in this conversation."));
      out.replaceChildren(h("pre", { class: "s-memory-notes", text: r.notes }), h("div", {}, h("button", { class: "btn btn-secondary", onclick: async () => {
        if (!(await confirmDialog("Clear these notes?", "The assistant loses its scratchpad for this conversation.", { confirm: "Clear", danger: true }))) return;
        try { await api.del(`/api/sessions/${sid}/notes`); out.replaceChildren(h("div", { class: "l-state" }, "No notes in this conversation.")); } catch (e) { toast(e.detail || "Could not clear the notes"); }
      } }, "Clear notes…")));
    }).then(() => body.append(out));
  }

  // ---------------------------------------------------------------- web cache
  async function cache() {
    const box = h("div", { class: "l-list" });
    body.append(box);
    box.replaceChildren(loading("Loading the cache…"));
    let d;
    try { d = await api.get("/api/memory/web-cache"); } catch (e) { return box.replaceChildren(failed("The web cache", e, () => show("cache"))); }
    if (!alive || tab !== "cache") return;
    const head = h("div", { class: "s-memory-between" }, h("span", { class: "s-memory-lede", text: "Searches kept so repeat questions don’t hit the web." }),
      h("button", { class: "btn btn-secondary", disabled: !d.queries.length && !d.pages.length, onclick: async () => {
        if (!(await confirmDialog("Clear the web cache?", "Saved searches and pages are deleted; the next question searches the web again.", { confirm: "Clear", danger: true }))) return;
        try { await api.del("/api/memory/web-cache"); toast("Web cache cleared"); show("cache"); } catch (e) { toast(e.detail || "Could not clear the cache"); }
      } }, "Clear cache…"));
    if (!d.queries.length && !d.pages.length) return box.replaceChildren(head, h("div", { class: "l-state" }, "Nothing cached yet."));
    box.replaceChildren(head,
      ...d.queries.map((c) => h("div", { class: "s-memory-row" }, h("div", { class: "s-memory-meta" }, icon("globe", 16), h("strong", { class: "s-memory-q", text: c.query }),
        c.expires_in > 0 ? tag("ok", "check", `expires in ${span(c.expires_in)}`) : tag("warn", "clock", "expired · refetch")),
        h("span", { class: "s-memory-dim", text: `${fmtDate(c.searched_at)} · ${c.sources} source${c.sources === 1 ? "" : "s"}` }))),
      d.pages.length ? h("h5", { text: "Pages read" }) : null,
      ...d.pages.map((p) => h("div", { class: "s-memory-row" }, h("strong", { class: "s-memory-q", text: p.title || p.url }), h("span", { class: "s-memory-dim s-memory-url", text: `${p.url} · fetched ${span(p.age)} ago` }))));
  }

  // ---------------------------------------------------------------- danger zone
  function danger() {
    const wrap = h("div", { class: "s-memory-danger" });
    body.append(wrap);
    const bar = h("i"), line = h("span", { class: "mono" }), barBox = h("div", { class: "l-progress hidden", role: "progressbar", "aria-label": "Re-embed progress" }, bar);
    const go = h("button", { class: "btn btn-secondary", onclick: reembed }, icon("refresh", 15), "Re-embed memory…");
    wrap.append(
      h("div", { class: "s-memory-row" }, h("div", { class: "s-memory-meta" }, icon("refresh", 18), h("strong", { text: "Re-embed memory" })),
        h("span", { class: "s-memory-note", text: "Rebuilds the search index. Long job; chat still works." }), barBox, line, h("div", {}, go)),
      h("div", { class: "s-memory-row s-memory-row-danger" }, h("div", { class: "s-memory-meta s-memory-bad" }, icon("danger", 18), h("strong", { text: "Delete all chats" })),
        h("span", { class: "s-memory-note", text: "Removes every session and message. Memory facts stay. This cannot be undone." }),
        h("div", {}, h("button", { class: "btn danger s-memory-del", onclick: async () => {
          if (!(await confirmDialog("Delete all chats?", "Every session and message is removed. Memory facts stay. This cannot be undone.", { confirm: "Delete all chats", danger: true }))) return;
          try { await api.del("/api/sessions"); ctx.shell.loadSessions(); toast("All chats deleted"); } catch (e) { toast(e.detail || "Could not delete the chats"); }
        } }, "Delete all chats…"))));

    let timer = null;
    ctx.onCleanup(() => clearInterval(timer));
    async function poll() {
      let s;
      try { s = await api.get("/api/memory/reembed/status"); } catch (_) { return; }
      if (!alive || tab !== "danger") return clearInterval(timer);
      const running = s.state === "running", pct = s.total ? Math.round((s.done / s.total) * 100) : 0;
      go.disabled = running;
      barBox.classList.toggle("hidden", !running);
      barBox.setAttribute("aria-valuenow", String(pct)); bar.style.width = `${pct}%`;
      line.textContent = running ? `running · ${pct} %${s.eta_seconds ? ` · about ${span(s.eta_seconds)} left` : ""}`
        : s.state === "done" ? `finished · ${s.done} of ${s.total} items re-embedded` : s.state === "error" ? `failed: ${s.error}` : "";
      if (running && !timer) timer = setInterval(poll, 2000);
      if (!running && timer) { clearInterval(timer); timer = null; }
    }
    async function reembed() {
      if (!(await confirmDialog("Re-embed all memory?", "Every stored item is embedded again with the current embedding model. This can take several minutes and uses the GPU.", { confirm: "Start" }))) return;
      try { await api.post("/api/memory/reembed"); } catch (e) { return toast(e.detail || "Could not start the job"); }
      poll();
    }
    poll();                                                  // also resumes the display of a job started earlier
  }

  show(tab);
}
