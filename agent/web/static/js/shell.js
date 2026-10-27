// The application frame (design L-Shell): desktop = sidebar + header + main (+ right inspector); phone = header +
// main + 4-tab bar, inspector as a bottom sheet. One DOM, adapted by CSS media queries (breakpoint 820 px).
import { h, relTime, clear } from "./dom.js";
import { icon } from "./icons.js";
import { mark } from "./mark.js";
import { api } from "./api.js";
import { state, set, subscribe, effectiveTheme, applyTheme } from "./store.js";
import { dialog, confirmDialog, toast } from "./ui.js";

// Destinations in the order of the design (L-Shell). `audit` and `prompt` have their own screens and stay reachable here
// until "Tools & audit" absorbs the audit log.
export const NAV = [
  ["chat", "Chat", "chat"], ["sessions", "Sessions & chat tools", "list"], ["board", "Board", "grid"], ["memory", "Memory", "memory"], ["knowledge", "Knowledge", "book"],
  ["skills", "Skills", "sparkle"], ["projects", "Projects & folder access", "folder"], ["code", "Code agent", "code"],
  ["whatsapp", "WhatsApp", "wa"], ["apps", "Apps", "download"], ["research", "Research, trust & web", "globe"], ["tools", "Tools & audit", "shield-check"],
  ["models", "Models & GPU", "cpu"], ["status", "Status", "chart"], ["settings", "Settings", "gear"], ["ops", "Operations", "sliders"],
  ["audit", "Audit log", "list"], ["prompt", "Prompt studio", "sparkle"],
];
const TABS = [["chat", "Chat", "chat"], ["board", "Board", "grid"], ["memory", "Memory", "memory"], ["more", "More", "menu"]];
export const TITLES = Object.fromEntries([...NAV.map(([k, l]) => [k, l]), ["system", "Connection"], ["patterns", "Patterns"], ["more", "More"]]);

/** Add the navigation entries of optional features (from /api/features: { key, label, nav: { icon, after } }). Returns the new keys. */
export function addNavFeatures(list) {
  const added = [];
  for (const f of list) {
    if (!f.nav || NAV.some(([k]) => k === f.key)) continue;
    const at = NAV.findIndex(([k]) => k === f.nav.after);
    NAV.splice(at < 0 ? NAV.length : at + 1, 0, [f.key, f.label, f.nav.icon]);
    TITLES[f.key] = f.label;
    added.push(f.key);
  }
  return added;
}

export function createShell(root) {
  let route = { name: "chat", params: {} };
  let sessions = [];

  const sessionList = h("div", { class: "l-sessions" });
  const nav = h("nav", { class: "l-nav", "aria-label": "Destinations" });
  const title = h("h3", { text: "Chat" });
  const ctxBox = h("span", { class: "l-ctx hidden", title: "Context window" });
  const modelPill = h("span", { class: "l-pill v-neutral" });
  const hostPill = h("span", { class: "l-pill v-neutral" });
  const themeBtn = h("button", { class: "btn btn-ghost btn-icon", "aria-label": "Toggle theme", title: "Toggle theme", onclick: () => {
    set({ theme: effectiveTheme() === "dark" ? "light" : "dark" });
  } });
  const ctxStrip = h("div", { class: "l-ctxstrip only-phone hidden" });
  const outlet = h("main", { class: "l-main", id: "main" });
  const tabs = h("nav", { class: "l-tabs", "aria-label": "Primary" });
  const inspector = h("aside", { class: "l-inspector hidden", "aria-label": "Evidence" });
  const banner = h("div", { class: "l-state v-warn hidden", role: "status", style: { borderRadius: 0, flexDirection: "row", alignItems: "center", gap: "8px" } });

  const side = h("aside", { class: "l-side only-desktop-flex" },
    h("div", { class: "l-brand" }, mark(26)),
    h("div", { class: "l-sidebtns" },
      h("button", { class: "btn btn-primary", onclick: () => newChat() }, icon("plus", 16), "New chat"),
      h("button", { class: "btn btn-secondary btn-icon", "aria-label": "Search sessions, Ctrl+K", title: "Search · Ctrl K", onclick: () => searchSessions() }, icon("search", 16))),
    h("div", { style: { minHeight: 0, overflow: "hidden", display: "flex", flexDirection: "column", gap: "2px" } }, h("div", { class: "l-sessions-h", text: "Sessions" }), sessionList),
    nav);
  const frame = h("div", { class: "l-frame" },
    h("header", { class: "l-head" }, title, ctxBox, modelPill, hostPill, themeBtn), ctxStrip, banner, outlet, tabs);
  root.replaceChildren(side, frame, inspector);

  // -------------------------------------------------------------- navigation
  const link = (href, cls, kids, label) => h("a", { href, class: cls, "aria-label": label, "aria-current": null }, ...kids);
  function drawNav() {
    nav.replaceChildren(...NAV.map(([k, label, ic]) => { const a = link("#/" + k, route.name === k ? "on" : "", [icon(ic, 17), h("span", { text: label })]); if (route.name === k) a.setAttribute("aria-current", "page"); return a; }));
    const moreSet = !["chat", "board", "memory"].includes(route.name);
    tabs.replaceChildren(...TABS.map(([k, label, ic]) => { const on = k === "more" ? moreSet : route.name === k; const a = link("#/" + k, "l-tab" + (on ? " on" : ""), [icon(ic, 22), h("span", { text: label })]); if (on) a.setAttribute("aria-current", "page"); return a; }));
  }

  // -------------------------------------------------------------- sessions
  async function loadSessions() {
    try { sessions = await api.get("/api/sessions"); } catch (_) { return; }
    drawSessions();
  }
  function drawSessions() {
    const current = route.name === "chat" ? Number(route.params.id) : null;
    const list = sessions.slice().sort((a, b) => b.id - a.id).slice(0, 8);          // the rest is one Ctrl+K away
    sessionList.replaceChildren(...list.map((s) => h("div", { style: { display: "flex", alignItems: "center" } },
      h("button", { class: "l-session" + (s.id === current ? " on" : ""), onclick: () => { location.hash = "#/chat/" + s.id; }, title: s.title || "New chat" },
        h("span", { class: "t", text: s.title || "New chat" }), h("span", { class: "when", text: relTime(s.started_at) })),
      h("button", { class: "l-ib sm", "aria-label": "Delete chat", title: "Delete chat", style: { opacity: .6 }, onclick: async (e) => {
        e.stopPropagation();
        if (!(await confirmDialog("Delete this chat?", "The messages are removed. Facts learned from it stay in memory.", { confirm: "Delete", danger: true }))) return;
        try { await api.del("/api/sessions/" + s.id); } catch (er) { return toast(er.detail || "Could not delete"); }
        if (current === s.id) location.hash = "#/chat";
        loadSessions();
      } }, icon("x", 14)))));
  }
  async function newChat() {
    try { const s = await api.post("/api/sessions"); await loadSessions(); location.hash = "#/chat/" + s.id; }
    catch (e) { toast(e.detail || "Could not start a chat"); }
  }
  async function searchSessions() {
    const input = h("input", { class: "input", placeholder: "Search chats…", "aria-label": "Search chats", autocomplete: "off" });
    const list = h("div", { class: "l-list", style: { maxHeight: "320px", overflow: "auto" } });
    const draw = () => list.replaceChildren(...sessions.filter((s) => (s.title || "New chat").toLowerCase().includes(input.value.toLowerCase())).slice().sort((a, b) => b.id - a.id).slice(0, 30)
      .map((s) => h("button", { class: "l-session", onclick: () => { location.hash = "#/chat/" + s.id; document.querySelector(".l-dialog-backdrop")?.remove(); } }, h("span", { class: "t", text: s.title || "New chat" }), h("span", { class: "when", text: relTime(s.started_at) }))));
    input.addEventListener("input", draw); draw();
    dialog({ title: "Search chats", body: h("div", { class: "l-list" }, input, list), actions: [{ id: "x", label: "Close" }] });
    input.focus();
  }
  document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); searchSessions(); } });

  // -------------------------------------------------------------- header
  const SHAPE = { ok: "", warn: "tri", danger: "sq" };
  /** A status pill: shape + long text on desktop; on phone the short text (model) or just the shape (host), as in the design. */
  function pill(el, variant, text, short = text, shape = SHAPE[variant] ?? "") {
    el.className = `l-pill v-${variant}`;
    el.setAttribute("aria-label", text);
    el.replaceChildren(h("span", { class: "shape " + shape, "aria-hidden": "true" }), h("span", { class: "long", text }), h("span", { class: "short", text: short }));
  }
  const shortModel = (name, st, detail) => st === "loading" ? (detail || "…") : st === "error" ? "offline" : ((name.split(":")[1] || name).split("-")[0] || name).slice(0, 8);
  function drawHeader() {
    const m = state.model, hst = state.host;
    const mv = m.state === "loaded" ? "ok" : m.state === "loading" ? "warn" : m.state === "error" ? "danger" : "neutral";
    const name = m.name || "model";
    const mText = m.state === "loaded" ? `${name} · loaded` : m.state === "loading" ? `${name} · loading${m.detail ? " " + m.detail : ""}` : m.state === "error" ? "no model server" : `${name} · idle`;
    pill(modelPill, mv, mText, shortModel(name, m.state, m.detail));
    const hv = hst.state === "asleep" ? "danger" : ["paused", "busy"].includes(hst.state) ? "warn" : hst.state === "ok" ? "ok" : "neutral";
    pill(hostPill, hv, hst.text || (hv === "ok" ? "host ok" : hv === "neutral" ? "host …" : hv === "danger" ? "PC asleep" : hst.state === "paused" ? "paused: gaming" : "host busy"), "");
    hostPill.classList.add("host");
    const dark = effectiveTheme() === "dark";
    themeBtn.replaceChildren(icon(dark ? "sun" : "moon", 18));
    drawCtx();
    banner.classList.toggle("hidden", state.connection !== "offline");
    if (state.connection === "offline") banner.replaceChildren(icon("wifioff", 16), "Offline — can’t reach the server. Messages can’t be sent until it is back.");
  }
  function drawCtx() {
    const c = state.ctx, show = route.name === "chat" && c;
    ctxBox.classList.toggle("hidden", !show); ctxStrip.classList.toggle("hidden", !show);
    if (!show) return;
    const pct = c.percent, v = pct >= 90 ? "danger" : pct >= 70 ? "warn" : "info";
    const k = (n) => Math.round(n / 1000) + " k";
    ctxBox.className = `l-ctx v-${v}`;
    ctxBox.replaceChildren(icon(pct >= 70 ? "warn" : "chart", 14), h("span", { text: `context ${pct} % · ${k(c.used)} / ${k(c.limit)}` }), h("span", { class: "bar", "aria-hidden": "true" }, h("i", { style: { width: Math.min(pct, 100) + "%", background: `var(--${v})` } })));
    ctxStrip.replaceChildren(icon(pct >= 70 ? "warn" : "chart", 13), h("span", { text: `context ${pct} %` }), h("span", { class: "bar", "aria-hidden": "true" }, h("i", { style: { width: Math.min(pct, 100) + "%", background: `var(--${v})` } })));
  }
  subscribe((s, patch) => { if ("theme" in patch || "density" in patch) applyTheme(); drawHeader(); });

  // -------------------------------------------------------------- status polling (model + host)
  async function pollStatus() {
    if (["unauthorized", "expired"].includes(state.connection)) return;
    try {
      const st = await api.get("/api/status");
      const loaded = (st.loaded_models || []).some((m) => m === st.chat_model || m.split(":")[0] === String(st.chat_model).split(":")[0]);
      set({ model: { name: st.chat_model, state: loaded ? "loaded" : "unloaded", detail: "" },
        host: { state: st.paused_reason ? "paused" : "ok", text: st.paused_reason ? "paused: " + st.paused_reason : "" } });
    } catch (_) { /* connection state already set by api.js */ }
  }
  setInterval(pollStatus, 20000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) pollStatus(); });

  // -------------------------------------------------------------- inspector
  function openInspector(content, { mode } = {}) {
    inspector.classList.remove("hidden");
    inspector.replaceChildren(h("div", { class: "sheet" }, content));
    inspector.onclick = (e) => { if (e.target === inspector && matchMedia("(max-width: 820px)").matches) closeInspector(); };
  }
  const closeInspector = () => { inspector.classList.add("hidden"); inspector.replaceChildren(); };

  const api_ = {
    outlet, setRoute(r) { route = r; title.textContent = r.title || TITLES[r.name] || r.name; drawNav(); drawSessions(); drawHeader(); closeInspector(); },
    setTitle(t) { title.textContent = t; }, loadSessions, openInspector, closeInspector, isPhone: () => matchMedia("(max-width: 820px)").matches, redraw: drawHeader, pollStatus, newChat,
    get sessions() { return sessions; },
  };
  drawNav(); drawHeader(); loadSessions(); pollStatus();
  return api_;
}
