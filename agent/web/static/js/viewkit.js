// viewkit — the generic, data-driven screen renderer (design S-View). A screen is declared as DATA: tabs of sections. The same
// section objects the design uses (js/views-data/*.js, built with the helpers from views-data/_h.js) render to DOM, and every
// section can additionally carry callbacks / live data so the screen is bound to the server instead of mock values.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
//  API
//
//    import { renderView, mountView, G, R, T, J, S, B, tab } from "../viewkit.js";
//
//    mountView(root, ctx, view)           // inside a screen's mount(): loads viewkit.css, sets the header title, cleans up on leave
//    const h = renderView(container, view, opts?)   // lower level; returns a handle (below)
//
//    view = { title, id?, tab?, notice?: banner section shown above every tab (e.g. "preview data"), tabs: [tab] }
//    tab  = { k, l,                       // key + label
//             s: [section...]             // static sections  ...or...
//             load: async (vk) => [section...]   // live sections; throw => error box with "Try again"; vk = { reload, signal }
//             poll?: ms,                  // silent reload every ms (skipped while the user is typing in the tab)
//             needs?: [string]            // "Needs backend" notes (design hand-off); omit once the tab is wired
//             empty?: string }            // text shown when no section is returned
//
//    handle = { el, tab(), setTab(k), reload({silent}), destroy() }      (destroy() is also run by mountView on navigation)
//
//  Section kinds (k) and their fields. Strings with a prefix are the design's short form; objects are the live form.
//  Buttons ("acts"): "Label", "+Label" (primary), "-Label" (danger) or { l, primary, danger, icon, disabled, onClick(ctx) };
//  click handlers may be async: the button shows busy and errors become a toast. A string button falls back to section.onAct(label, ctx).
//  Without any handler a click says "Demo data, not connected yet". ctx = { ev, section, item?, index?, reload }.
//  Table/list tags and cells: "ok:text" "warn:" "danger:" "info:" "neutral:" "block:" (icon + word pill), "m:text" (mono),
//  or an object { t, tone, icon, mono, btn:true, onClick }.
//
//    group    { title, rows:[row], onChange?(value,row), p? }
//             row = { label, help, ctl, val, def, f, onChange?(value,row), onReset?(row), onBrowse?(cur), opts?, min, max, step, unit,
//                     changed?, restart?, error?, disabled? }
//             ctl: "sw" (val true/false or "on"/"off") | "tx" | "nm" | "pa" (path + Browse) | "ta" | "sl" (range) | "seg:A|B" | "sel:A|B".
//             opts may be given as an array instead of after the colon; items are strings or { v, l }.
//             f (flags, space separated): c = changed, r = restart required, p = PROPOSAL, e:message = error.
//             onChange may be async and return { error } / { val } or throw: the message shows under the row. "changed" and "Reset"
//             follow val vs def automatically once the user touches the row.
//    banner   { v: info|warn|danger|ok|block, title, text, acts }
//    job      { title, sub, pct, st: running|done|failed|queued|paused, el, eta, log:[string], acts, poll?: async()=>patch, pollMs? }
//    table    { title, cols:[string], rows:[[cell]] | [{cells, onClick}], acts, onAct?, p? }
//    ld       { title, items:[[title, sub, tag] | {t, sub, tag, ...}], detail:{title, rows:[[k,v]], acts}, detailFor?(item,i) -> detail
//               (may be async), onSelect?(item,i), sel? }
//    cards    { title, items:[{t, s, tag, img, src, sw, p, onToggle(on), onClick}], acts }
//    log      { title, lines:["I: text" | "W: .." | "E: .." | {lv:"I"|"W"|"E", t, w?}], poll?: async()=>({lines}), pollMs? }
//    wizard   { steps:[label], active, fields:[row], fieldsFor?(step), summary: string | (rows)=>string, onNext?(step), onBack?(step),
//               onFinish?(), finish? }
//    danger   { title, text, word, btn, onConfirm() }         (button enabled only when the word is typed exactly)
//    empty    { title, text, cmd }                            (cmd gets a copy button)
//    stats    { title?, tiles:[[label, value, sub]] }          curve { title?, pts:[number], sub }
//    filters  { fields:[[label, value, icon] | {l, v, i, input, placeholder, options, onInput(value), onClick}] }
//    actions  { items:[[label, icon, primary] | {l, i, primary, onClick}] }
//    think    { mode, detail, sec?, code? }                   block { kind } demo or { block } = a real server block (renderBlock)
//    approval { level } demo or { call, onDecide(bool) } = a real approval (see components/approval.js)
//    bars     { title?, items:[[label, percent, text]] }      test { title?, cmd, tier, why, onTest?(cmd) -> {tier, why} }
//    text     { title?, text }   prose { scene, paras, choices, onChoose?(c) }   chat { title?, msgs:[[who,text]], onSend?(text), sendLabel? }
//    custom   { title?, render(section, ctx) -> Node }          a screen's own widget inside a tab
//    qr       { addr, src? }     ui {}  (the component sampler)
//  Any section may add  poll: async () => patch, pollMs  to refresh itself in place (job and log stop on their own when finished).
//  Any section may carry  p: true  to show the design's PROPOSAL tag.
//
//  Example (a screen bound to live data):
//
//    export async function mount(root, ctx) {
//      mountView(root, ctx, { title: "Operations", tabs: [{
//        k: "worker", l: "Background worker", poll: 5000,
//        load: async () => { const w = await api.get("/api/worker"); return [
//          S([["State", w.state, ""], ["Queue", String(w.queued), "jobs"]]),
//          G("Worker", [{ label: "Enabled", ctl: "sw", val: w.enabled, def: true, onChange: (v) => api.put("/api/worker", { enabled: v }) }]) ]; },
//      }] });
//    }
// ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
import { h, svg } from "./dom.js";
import { icon } from "./icons.js";
import { copyText, toast } from "./ui.js";
import { renderBlock } from "./components/block.js";
import { approvalCard } from "./components/approval.js";
import { thinking, DEMO_CODE } from "./components/thinking.js";
import { loadingBox, errorBox, emptyBox } from "./components/ops-ui.js";

export * from "./views-data/_h.js";

// ------------------------------------------------------------------------------------------------ small parts
const TONE_ICON = { ok: "check", warn: "warn", danger: "danger", info: "info", neutral: "shield-q", block: "blocked" };
const JOB_STATE = { running: ["info", "Running", "clock"], done: ["ok", "Done", "check"], failed: ["danger", "Failed", "x"], queued: ["neutral", "Queued", "clock"], paused: ["warn", "Paused", "clock"] };
const NOT_WIRED = () => toast("Demo data, not connected yet");
const errText = (e) => (e && (e.detail || e.message)) || String(e || "Failed");

/** icon + word on a semantic background, border style carries meaning too (neutral = dashed). */
export const pill = (tone, iconName, text) => h("span", { class: `vk-pill v-${tone}` + (tone === "neutral" ? " dashed" : tone === "block" ? " thick" : "") }, iconName ? icon(iconName, 13) : null, text);

/** "ok:text" / "m:text" / plain text / cell object -> normalised { t, tone?, mono?, ... } */
export function cellOf(c) {
  if (c && typeof c === "object" && !Array.isArray(c)) return c;
  const m = /^(ok|warn|danger|info|neutral|block|m):([\s\S]*)$/.exec(String(c));
  if (!m) return { t: String(c) };
  return m[1] === "m" ? { t: m[2], mono: true } : { t: m[2], tone: m[1] };
}
function cellEl(c, ctx) {
  const o = cellOf(c);
  if (o.tone) return pill(o.tone, o.icon || TONE_ICON[o.tone], o.t);
  if (o.btn) return btn({ l: o.t, onClick: o.onClick }, ctx, "btn-ghost");
  return h("span", { class: o.mono ? "vk-mono" : "vk-cell", text: o.t });
}

/** "+Label" primary, "-Label" danger, "Label" secondary; or an object. */
export function btnSpec(x) {
  if (x && typeof x === "object") return { ...x, l: x.l ?? x.label };
  const s = String(x), c = s[0];
  return c === "+" ? { l: s.slice(1), primary: true } : c === "-" ? { l: s.slice(1), danger: true } : { l: s };
}
async function runBusy(b, fn) {
  b.disabled = true; b.setAttribute("aria-busy", "true");
  try { await fn(); } catch (e) { toast(errText(e)); } finally { b.disabled = !!b._off; b.removeAttribute("aria-busy"); }
}
/** One button. ctx supplies { section, onAct } fallbacks; extra is merged into the click context (item, index...). */
function btn(spec, ctx, cls = "", extra = {}) {
  const s = btnSpec(spec);
  const b = h("button", { type: "button", class: `btn ${cls || (s.primary ? "btn-primary" : "btn-secondary")} vk-btn${s.danger ? " danger-o" : ""}`, disabled: !!s.disabled },
    s.icon ? icon(s.icon, 15) : null, s.l);
  b._off = !!s.disabled;
  b.addEventListener("click", (ev) => {
    const c = { ev, section: ctx && ctx.section, reload: ctx && ctx.reload, ...extra };
    const fn = s.onClick ? () => s.onClick(c) : ctx && ctx.onAct ? () => ctx.onAct(s.l, c) : null;
    if (!fn) return NOT_WIRED();
    runBusy(b, fn);
  });
  return b;
}
const btns = (list, ctx, cls) => (list || []).map((a) => btn(a, ctx, cls));

// ------------------------------------------------------------------------------------------------ settings rows (group + wizard)
const parseFlags = (f = "") => {
  const em = /e:(.*)$/.exec(f), fl = f.replace(/e:.*$/, "");
  const has = (x) => new RegExp("(^|\\s)" + x + "(\\s|$)").test(fl);
  return { c: has("c"), r: has("r"), p: has("p"), e: em ? em[1].trim() : "" };
};
const optList = (r) => {
  const raw = r.opts || (String(r.ctl).includes(":") ? String(r.ctl).slice(String(r.ctl).indexOf(":") + 1).split("|") : []);
  return raw.map((o) => (typeof o === "object" ? o : { v: o, l: o }));
};
const asBool = (v) => v === true || v === "on" || v === "true" || v === 1;
const defaultVal = (r, ctl) => (ctl === "sw" ? asBool(r.def) : /^\(.*\)$/.test(String(r.def ?? "")) || r.def == null ? "" : r.def);
const sameVal = (a, b, ctl) => (ctl === "sw" ? asBool(a) === asBool(b) : String(a ?? "").replace(/\s/g, "") === String(b ?? "").replace(/\s/g, ""));

/** One settings row; `owner` supplies the fallback onChange. Returns the element (the row object's `val` is kept current). */
function settingRow(r, owner, ctx) {
  const ctl = String(r.ctl || "tx").split(":")[0], fl = parseFlags(r.f), opts = optList(r);
  let touched = false, error = fl.e, busy = false;
  if (ctl === "sw") r.val = asBool(r.val);
  const label = h("strong", { text: r.label }), help = r.help ? h("div", { class: "vk-help", text: r.help }) : null;
  const meta = h("div", { class: "vk-meta" }), err = h("div", { class: "vk-err", role: "alert" });
  const ctrl = h("div", { class: "vk-ctl" });
  const isChanged = () => (r.changed !== undefined ? !!r.changed : touched ? !sameVal(r.val, defaultVal(r, ctl), ctl) : fl.c);
  const drawMeta = () => {
    const ch = isChanged();
    meta.replaceChildren(h("span", { text: "default " + (r.def === "" || r.def == null ? "—" : r.def) }),
      ch ? h("span", { class: "vk-changed", text: "changed" }) : null,
      ch ? h("button", { type: "button", class: "btn btn-ghost vk-reset", onclick: reset }, "Reset") : null);
    err.replaceChildren(...(error ? [icon("danger", 14), error] : []));
    err.hidden = !error;
    ctrl.classList.toggle("bad", !!error);
  };
  async function commit(v) {
    r.val = v; touched = true; error = ""; drawMeta();
    const fn = r.onChange || owner.onChange;
    if (!fn) return;
    busy = true; ctrl.classList.add("busy");
    try {
      const res = await fn(v, r);
      if (res && res.error) throw new Error(res.error);
      if (res && "val" in res) { r.val = res.val; draw(); }
    } catch (e) { error = errText(e); }
    busy = false; ctrl.classList.remove("busy"); drawMeta();
  }
  async function reset() {
    const d = defaultVal(r, ctl);
    if (r.onReset) { try { const res = await r.onReset(r); if (res && "val" in res) r.val = res.val; else r.val = d; } catch (e) { error = errText(e); } touched = true; r.changed = undefined; draw(); drawMeta(); return; }
    r.val = d; r.changed = undefined; draw(); await commit(d);
  }
  function draw() {
    const dis = !!r.disabled;
    let node;
    if (ctl === "sw") {
      node = h("button", { type: "button", role: "switch", class: "vk-switch", "aria-checked": String(!!r.val), "aria-label": r.label, disabled: dis },
        h("span", { class: "track" }, h("i")), h("span", { text: r.val ? "On" : "Off" }));
      node.addEventListener("click", () => { r.val = !r.val; node.setAttribute("aria-checked", String(r.val)); node.lastChild.textContent = r.val ? "On" : "Off"; commit(r.val); });
    } else if (ctl === "ta") {
      node = h("textarea", { class: "input vk-ta", "aria-label": r.label, disabled: dis }); node.value = r.val ?? "";
      node.addEventListener("change", () => commit(node.value));
    } else if (ctl === "sl") {
      const unit = r.unit ?? "%", out = h("span", { class: "vk-mono", text: `${r.val} ${unit}`.trim() });
      const rng = h("input", { type: "range", class: "vk-range", min: r.min ?? 10, max: r.max ?? 100, step: r.step ?? 1, "aria-label": r.label, disabled: dis }); rng.value = parseFloat(r.val) || 0;
      rng.addEventListener("input", () => { out.textContent = `${rng.value} ${unit}`.trim(); });
      rng.addEventListener("change", () => commit(Number(rng.value)));
      node = h("div", { class: "vk-slider" }, rng, out);
    } else if (ctl === "seg") {
      node = h("div", { class: "vk-seg", role: "radiogroup", "aria-label": r.label }, opts.map((o) => {
        const on = String(o.v) === String(r.val);
        return h("button", { type: "button", role: "radio", "aria-checked": String(on), class: on ? "on" : "", disabled: dis,
          onclick: () => { if (String(o.v) === String(r.val)) return; r.val = o.v; draw(); commit(o.v); } }, on ? icon("check", 13) : null, o.l);
      }));
    } else if (ctl === "sel") {
      node = h("select", { class: "input vk-sel", "aria-label": r.label, disabled: dis }, opts.map((o) => h("option", { value: o.v, text: o.l, selected: String(o.v) === String(r.val) })));
      node.addEventListener("change", () => commit(node.value));
    } else {                                                    // tx, nm, pa
      const inp = h("input", { class: "input vk-in" + (ctl === "nm" || ctl === "pa" ? " mono-in" : ""), "aria-label": r.label, disabled: dis, inputmode: ctl === "nm" ? "decimal" : null }); inp.value = r.val ?? "";
      inp.addEventListener("change", () => commit(ctl === "nm" && typeof r.def === "number" && inp.value.trim() !== "" && !isNaN(Number(inp.value)) ? Number(inp.value) : inp.value));
      node = ctl === "pa" ? h("div", { class: "vk-path" }, inp, h("button", { type: "button", class: "btn btn-secondary", disabled: dis, onclick: async () => {
        if (!r.onBrowse) return inp.focus();
        const p = await r.onBrowse(inp.value); if (p != null) { inp.value = p; commit(p); }
      } }, "Browse")) : inp;
    }
    ctrl.replaceChildren(node);
    ctrl.classList.toggle("busy", busy);
  }
  draw(); drawMeta();
  return h("div", { class: "vk-row" },
    h("div", { class: "vk-rowl" }, h("div", { class: "vk-rowt" }, label,
      (r.restart ?? fl.r) ? h("span", { class: "vk-restart" }, icon("refresh", 11), "restart required") : null,
      (r.p ?? fl.p) ? h("span", { class: "vk-ptag", text: "PROPOSAL" }) : null), help, meta, err), ctrl);
}

// ------------------------------------------------------------------------------------------------ section renderers
const TITLED = new Set(["group", "table", "cards"]);
const TITLED_IF = new Set(["ld", "curve", "chat", "bars", "stats", "text", "custom"]);

const R = {
  group(s, ctx) {
    ctx = { ...ctx, section: s };
    const rows = (s.rows || []).map((r) => settingRow(r, s, ctx));
    return h("div", { class: "vk-card" }, rows.length ? rows : h("div", { class: "vk-help", style: { padding: "14px 0" }, text: "Nothing to set here." }));
  },
  banner(s, ctx) {
    const v = s.v || "info", ic = { info: "info", warn: "warn", danger: "danger", ok: "check", block: "blocked" }[v] || "info";
    ctx = { ...ctx, section: s };
    return h("div", { role: "note", class: `vk-banner v-${v}${v === "block" ? " dashed" : ""}` }, icon(ic, 20),
      h("div", { class: "t" }, h("strong", { text: s.title }), h("span", { text: s.text })), btns(s.acts, ctx));
  },
  job(s, ctx) {
    ctx = { ...ctx, section: s };
    const m = JOB_STATE[s.st] || JOB_STATE.running;
    const bar = h("i", { style: { width: Math.max(0, Math.min(100, s.pct || 0)) + "%" }, class: s.st === "done" ? "ok" : "" });
    return h("div", { class: "vk-panel" },
      h("div", { class: "vk-jobhd" }, h("div", { class: "t" }, h("strong", { text: s.title }), h("div", { class: "vk-help", text: s.sub || "" })),
        s.p ? h("span", { class: "vk-ptag", text: "PROPOSAL" }) : null, pill(m[0], m[2], m[1])),
      h("div", { class: "vk-bar", role: "progressbar", "aria-valuenow": s.pct || 0, "aria-valuemin": 0, "aria-valuemax": 100, "aria-label": s.title }, bar),
      h("div", { class: "vk-jobmeta" }, h("span", { text: `${s.pct ?? 0} %` }), h("span", { text: `elapsed ${s.el ?? "—"}` }), h("span", { text: `ETA ${s.eta ?? "—"}` })),
      s.log && s.log.length ? h("div", { class: "vk-monobox" }, s.log.map((l) => h("div", { text: l }))) : null,
      s.acts === undefined ? btnRow(["Stop"], ctx) : (s.acts || []).length ? btnRow(s.acts, ctx) : null);
  },
  table(s, ctx) {
    ctx = { ...ctx, section: s };
    const cols = s.cols || [];
    const rows = (s.rows || []).map((r) => (Array.isArray(r) ? { cells: r } : r));
    return h("div", { class: "vk-panel tight" }, h("div", { class: "vk-scroll" }, h("table", { class: "vk-table", style: { minWidth: Math.max(480, cols.length * 130) + "px" } },
      h("thead", {}, h("tr", {}, cols.map((c) => h("th", { scope: "col", text: c })))),
      h("tbody", {}, rows.map((r) => {
        const tr = h("tr", { class: r.onClick ? "click" : "", tabindex: r.onClick ? 0 : null }, r.cells.map((c, j) => h("td", {}, cellEl(typeof c === "string" && j === r.cells.length - 1 && cols[j] === "" && !/^\w+:/.test(c) ? { t: c, btn: true, onClick: (cc) => (s.onAct ? s.onAct(c, { ...cc, row: r }) : NOT_WIRED()) } : c, { ...ctx, row: r }))));
        if (r.onClick) { tr.addEventListener("click", (ev) => { if (!ev.target.closest("button")) r.onClick({ row: r, section: s }); }); tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter") r.onClick({ row: r, section: s }); }); }
        return tr;
      })))));
  },
  ld(s, ctx) {
    ctx = { ...ctx, section: s };
    const items = (s.items || []).map((it) => (Array.isArray(it) ? { t: it[0], sub: it[1], tag: it[2] } : it));
    let sel = s.sel ?? 0, seq = 0;
    const list = h("div", { class: "vk-panel vk-list", role: "listbox", "aria-label": s.title || "List" }), det = h("div", { class: "vk-panel", "aria-live": "polite" });
    const drawList = () => list.replaceChildren(...(items.length ? items.map((it, i) => {
      const tag = it.tag ? cellOf(it.tag) : null;
      return h("div", { class: "vk-li" + (i === sel ? " on" : ""), role: "option", tabindex: 0, "aria-selected": String(i === sel), onclick: () => pick(i),
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(i); } } },
        h("div", { class: "t" }, h("div", { class: "n", text: it.t }), h("div", { class: "vk-help", text: it.sub || "" })),
        tag && tag.tone ? pill(tag.tone, tag.icon || TONE_ICON[tag.tone], tag.t) : null);
    }) : [h("div", { class: "vk-help", text: s.emptyText || "Nothing here yet." })]));
    const drawDet = async () => {
      const my = ++seq, it = items[sel];
      if (!it) return det.replaceChildren();
      let d = s.detail;
      if (s.detailFor) { det.replaceChildren(loadingBox()); try { d = await s.detailFor(it, sel); } catch (e) { if (my === seq) det.replaceChildren(errorBox(e, drawDet)); return; } if (my !== seq) return; }
      if (!d) return det.replaceChildren();
      det.replaceChildren(h("strong", { class: "vk-dt", text: d.title }), ...(d.rows || []).map((r) => h("div", { class: "vk-kv" }, h("span", { text: r[0] }), h("span", { text: r[1] }))),
        h("div", { class: "vk-btnrow" }, (d.acts || []).map((a) => btn(a, ctx, "", { item: it, index: sel }))));
    };
    function pick(i) { sel = i; drawList(); drawDet(); if (s.onSelect) s.onSelect(items[i], i); }
    drawList(); drawDet();
    return h("div", { class: "vk-ld" }, list, det);
  },
  cards(s, ctx) {
    ctx = { ...ctx, section: s };
    return h("div", { class: "vk-cards" }, (s.items || []).map((c) => {
      const t = c.tag ? cellOf(c.tag) : null;
      const art = h("article", { class: "vk-panel" + (c.onClick ? " click" : ""), tabindex: c.onClick ? 0 : null },
        c.src ? h("img", { class: "vk-img", src: c.src, alt: c.alt || c.t, loading: "lazy" }) : c.img ? h("div", { class: "vk-imgph", role: "img", "aria-label": "Placeholder" }, "image / cover") : null,
        h("div", { class: "vk-cardhd" }, h("strong", { text: c.t }), t && t.tone ? pill(t.tone, t.icon || TONE_ICON[t.tone], t.t) : null),
        h("span", { class: "vk-help", text: c.s || "" }), c.p ? h("span", { class: "vk-ptag", text: "PROPOSAL" }) : null,
        c.sw !== undefined ? switchBtn(asBool(c.sw), (on) => (c.onToggle ? c.onToggle(on, c) : NOT_WIRED())) : null);
      if (c.onClick) art.addEventListener("click", (e) => { if (!e.target.closest("button")) c.onClick(c); });
      return art;
    }));
  },
  log(s, ctx) {
    const lines = (s.lines || []).map((l) => {
      if (typeof l === "object") return l;
      const m = /^([IWE]):\s?([\s\S]*)$/.exec(l); return m ? { lv: m[1], t: m[2] } : { lv: "I", t: String(l) };
    });
    const LV = { I: ["INFO", "info"], W: ["WARN", "warn"], E: ["ERROR", "danger"] };
    const body = h("div", { class: "vk-logbody", tabindex: 0, role: "log", "aria-label": s.title || "Log" }, lines.map((l) => {
      const m = LV[l.lv] || LV.I;
      return h("div", { class: "ln" }, h("span", { class: "lv v-" + m[1], text: l.w || `[${m[0]}]` }), h("span", { text: l.t }));
    }));
    const follow = () => { body.scrollTop = body.scrollHeight; };
    if (s.follow !== false) queueMicrotask(follow);
    return h("div", { class: "vk-log" }, h("div", { class: "hd" }, h("span", { text: s.title || "" }),
      h("button", { type: "button", class: "btn btn-ghost vk-sm", onclick: follow }, icon("down", 13), "Follow"),
      h("button", { type: "button", class: "btn btn-ghost vk-sm", onclick: () => copyText(lines.map((l) => (l.w || `[${(LV[l.lv] || LV.I)[0]}]`) + " " + l.t).join("\n"), "Log copied") }, icon("copy", 13), "Copy")), body);
  },
  wizard(s, ctx) {
    ctx = { ...ctx, section: s };
    let active = s.active || 0;
    const root = h("div", { class: "vk-panel wide" });
    const draw = () => {
      const steps = s.steps || [], last = active >= steps.length - 1;
      const rows = (s.fieldsFor ? s.fieldsFor(active) : s.fields || []);
      const summary = typeof s.summary === "function" ? s.summary(rows) : s.summary;
      root.replaceChildren(
        h("div", { class: "vk-steps" }, steps.map((l, j) => h("span", { class: "vk-step" + (j === active ? " on" : "") + (j < active ? " done" : ""), "aria-current": j === active ? "step" : null },
          h("span", { class: "n", text: j < active ? "✓" : String(j + 1) }), l))),
        ...rows.map((r) => settingRow(r, s, ctx)),
        summary ? h("div", { class: "vk-summary" }, h("strong", { text: "Summary" }), h("br"), summary) : null,
        h("div", { class: "vk-btnrow" },
          h("button", { type: "button", class: "btn btn-secondary vk-btn", disabled: active === 0, onclick: async (e) => { if (s.onBack) await s.onBack(active); active = Math.max(0, active - 1); draw(); } }, "Back"),
          (() => { const b = h("button", { type: "button", class: "btn btn-primary vk-btn" }, last ? (s.finish || "Finish") : "Next");
            b.addEventListener("click", () => runBusy(b, async () => {
              if (last) { if (s.onFinish) await s.onFinish(); else NOT_WIRED(); return; }
              if (s.onNext && (await s.onNext(active)) === false) return;
              active += 1; draw();
            })); return b; })()));
    };
    draw();
    return root;
  },
  danger(s) {
    const input = h("input", { class: "input vk-in mono-in", placeholder: s.word, autocomplete: "off", id: "vk-d" + Math.random().toString(36).slice(2), "aria-label": `Type ${s.word}` });
    const b = h("button", { type: "button", class: "btn vk-dangerbtn", disabled: true }, s.btn);
    input.addEventListener("input", () => { b.disabled = input.value !== s.word; });
    b.addEventListener("click", () => (s.onConfirm ? runBusy(b, async () => { await s.onConfirm(); input.value = ""; b.disabled = true; }) : NOT_WIRED()));
    return h("div", { class: "vk-panel vk-dz" }, h("div", { class: "vk-dzhd" }, icon("danger", 20), h("strong", { text: s.title })), h("span", { text: s.text }),
      h("div", { class: "field" }, h("label", { for: input.id }, "Type ", h("strong", { class: "vk-mono", text: s.word }), " to enable the button"), input), h("div", {}, b));
  },
  empty(s) {
    return h("div", { class: "vk-panel start" }, h("strong", { class: "vk-big", text: s.title }), h("span", { class: "vk-help", text: s.text }),
      s.cmd ? h("div", { class: "vk-cmd" }, "$ " + s.cmd, h("button", { type: "button", "aria-label": "Copy command", onclick: () => copyText(s.cmd, "Command copied") }, icon("copy", 14))) : null);
  },
  stats(s) {
    return h("div", { class: "vk-stats" }, (s.tiles || []).map((t) => { const [l, v, sub] = Array.isArray(t) ? t : [t.l, t.v, t.sub];
      return h("div", { class: "vk-panel tile" }, h("span", { class: "k", text: l }), h("span", { class: "v", text: v }), h("span", { class: "vk-help", text: sub || "" })); }));
  },
  curve(s) {
    const pts = s.pts || [], mx = Math.max(...pts), mn = Math.min(...pts);
    const xy = pts.map((p, j) => [30 + j * (284 / Math.max(1, pts.length - 1)), 100 - ((p - mn) / (mx - mn || 1)) * 85 - 5]);
    const g = svg("svg", { viewBox: "0 0 320 120", width: "100%", role: "img", "aria-label": `${s.title || "Curve"} from ${mx} to ${pts[pts.length - 1]}`, class: "vk-curve" },
      svg("path", { d: "M30 10V100H314", fill: "none", stroke: "var(--color-divider)" }),
      svg("path", { d: "M" + xy.map((p) => p[0].toFixed(1) + " " + p[1].toFixed(1)).join("L"), fill: "none", stroke: "var(--series-1)", "stroke-width": 2.4 }),
      ...xy.filter((p, j) => j % 3 === 2).map((p) => svg("rect", { x: p[0] - 4, y: p[1] - 4, width: 8, height: 8, fill: "var(--color-bg)", stroke: "var(--series-1)", "stroke-width": 2 })));
    return h("div", { class: "vk-panel" }, g, h("span", { class: "vk-help", text: `${s.sub || ""} · squares = checkpoints` }));
  },
  filters(s) {
    return h("div", { class: "vk-filters" }, (s.fields || []).map((f) => {
      const o = Array.isArray(f) ? { l: f[0], v: f[1], i: f[2] } : f;
      const kids = [icon(o.i || "search", 15), h("span", { class: "l", text: o.l })];
      let el;
      if (o.options) { const sel = h("select", { class: "vk-fsel", "aria-label": o.l }, o.options.map((x) => h("option", { value: typeof x === "object" ? x.v : x, text: typeof x === "object" ? x.l : x, selected: String(typeof x === "object" ? x.v : x) === String(o.v) }))); sel.addEventListener("change", () => o.onInput && o.onInput(sel.value)); kids.push(sel); }
      else if (o.input) { let t; const inp = h("input", { class: "vk-fin", placeholder: o.placeholder || o.l, "aria-label": o.l }); inp.value = o.v ?? ""; inp.addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => o.onInput && o.onInput(inp.value), 250); }); kids.push(inp); }
      else kids.push(h("span", { class: "v", text: o.v }));
      el = h(o.onClick ? "button" : "div", { class: "vk-chip", type: o.onClick ? "button" : null, onclick: o.onClick ? (e) => o.onClick(e) : null }, kids);
      return el;
    }));
  },
  actions(s, ctx) {
    ctx = { ...ctx, section: s };
    return h("div", { class: "vk-btnrow" }, (s.items || []).map((a) => { const o = Array.isArray(a) ? { l: a[0], icon: a[1], primary: !!a[2] } : a; return btn(o, ctx); }));
  },
  think(s) { return thinking({ kind: s.mode, detail: s.detail || "", seconds: s.sec ?? 18, code: s.code ?? (s.mode === "coding" || s.mode === "editing" ? DEMO_CODE : null), file: s.detail }); },
  block(s) {
    if (s.block) return renderBlock(s.block);
    const demo = { diff: { kind: "diff", title: "Changes", provenance: "computed", spec: { files: [{ path: "rps/game.py", hunks: [{ lines: [{ t: "ctx", text: "def play(rounds):" }, { t: "del", text: "    return None" }, { t: "add", text: "    score = Counter()" }, { t: "add", text: "    return score" }] }] }] } } }[s.kind];
    return demo ? renderBlock(demo) : emptyBox("info", `Block "${s.kind}"`, "No sample for this kind; pass { block } with real data.");
  },
  approval(s) {
    const level = (s.level || "CAUTION").toUpperCase();
    const call = s.call || { approval_id: "demo-" + level.toLowerCase(), tool_name: "run_command", risk_level: level, arguments: { command: level === "BLOCKED" ? "write ~/.ssh/config" : "rm -r ~/Documents/old" },
      description: level === "BLOCKED" ? "Blocked by the safety policy: outside any allowed folder." : "Deletes files, cannot be undone." };
    return approvalCard(call, { status: s.status || "pending", onDecide: s.onDecide || NOT_WIRED });
  },
  bars(s) {
    return h("div", { class: "vk-panel vk-bars" }, (s.items || []).map((b, j) => h("div", { class: "r" }, h("span", { text: b[0] }),
      h("span", { class: "t", "aria-hidden": "true" }, h("i", { style: { width: Math.min(100, b[1]) + "%", background: `var(--series-${(j % 6) + 1})` } })), h("span", { class: "vk-mono", text: b[2] || String(b[1]) }))));
  },
  test(s, ctx) {
    const inp = h("input", { class: "input vk-in mono-in", "aria-label": "Command" }); inp.value = s.cmd || "";
    const res = h("div", { class: "vk-testres" });
    const show = (tier, why) => { const c = cellOf(tier); res.replaceChildren(c.tone ? pill(c.tone, TONE_ICON[c.tone], c.t) : c.t, h("span", { class: "vk-help", text: why || "" })); };
    if (s.tier) show(s.tier, s.why);
    const go = s.onTest ? h("button", { type: "button", class: "btn btn-secondary vk-btn", onclick: (e) => runBusy(e.currentTarget, async () => { const r = await s.onTest(inp.value); show(r.tier, r.why); }) }, "Test") : null;
    return h("div", { class: "vk-panel" }, h("label", { class: "vk-help", text: "Command" }), h("div", { class: "vk-path" }, inp, go), res);
  },
  text(s) { return h("p", { class: "vk-text", text: s.text }); },
  custom(s, ctx) { return s.render(s, ctx); },                 // escape hatch: render(section, ctx) -> Node (a screen's own widget)
  prose(s) {
    return h("div", { class: "vk-panel wide" }, h("span", { class: "vk-kicker", text: s.scene }), (s.paras || []).map((p) => h("p", { class: "vk-prose", text: p })),
      h("div", { class: "vk-btnrow" }, (s.choices || []).map((c) => h("button", { type: "button", class: "btn btn-secondary vk-btn pill", onclick: () => (s.onChoose ? s.onChoose(c) : NOT_WIRED()) }, c))));
  },
  chat(s) {
    const col = h("div", { class: "vk-panel vk-minichat" });
    const add = (who, t) => col.insertBefore(h("div", { class: "vk-msg " + (who === "you" ? "me" : "bot"), text: t }), col.querySelector("form"));
    (s.msgs || []).forEach((m) => col.append(h("div", { class: "vk-msg " + (m[0] === "you" ? "me" : "bot"), text: m[1] })));
    if (s.onSend) {
      const inp = h("input", { class: "input vk-in", placeholder: "Message", "aria-label": "Message" });
      col.append(h("form", { class: "vk-path", onsubmit: async (e) => { e.preventDefault(); const t = inp.value.trim(); if (!t) return; inp.value = ""; add("you", t);
        try { const r = await s.onSend(t); if (r) add("bot", r); } catch (er) { toast(errText(er)); } } }, inp, h("button", { class: "btn btn-primary", type: "submit" }, s.sendLabel || "Send")));
    }
    return col;
  },
  qr(s) {
    return h("div", { class: "vk-panel row" }, s.src ? h("img", { class: "vk-qr", src: s.src, alt: "QR code" }) : h("div", { class: "vk-qr ph", role: "img", "aria-label": "QR code placeholder" }),
      h("div", {}, h("div", { class: "vk-help", text: "Address" }), h("div", { class: "vk-addr", text: s.addr }), s.src ? null : h("div", { class: "vk-help", text: "QR code placeholder, generated by the app" })));
  },
  ui() {
    const chips = [["ok", "check", "Verified"], ["info", "info", "Likely"], ["warn", "warn", "Single source"], ["danger", "danger", "Conflicting"], ["block", "blocked", "Blocked"], ["neutral", "shield-q", "From memory"]];
    const card = (title, kids) => h("div", { class: "vk-panel" }, h("strong", { text: title }), kids);
    return h("div", { class: "vk-ui" },
      card("Buttons", h("div", { class: "vk-btnrow" }, h("button", { class: "btn btn-primary vk-btn" }, "Primary"), h("button", { class: "btn btn-secondary vk-btn" }, "Secondary"), h("button", { class: "btn btn-ghost vk-btn" }, "Ghost"),
        h("button", { class: "btn danger vk-btn" }, "Danger"), h("button", { class: "btn btn-primary vk-btn", disabled: true }, "Disabled"), h("button", { class: "btn btn-secondary btn-icon", "aria-label": "Pin", style: { width: "44px", height: "44px" } }, icon("pin", 18)))),
      card("Inputs", h("div", { class: "vk-inputs" }, h("div", { class: "field" }, h("label", { text: "Text" }), h("input", { class: "input vk-in", value: "Nightage" })),
        h("div", { class: "field" }, h("label", { text: "With error" }), h("input", { class: "input vk-in bad", value: "96" }), h("span", { class: "vk-err" }, icon("danger", 14), "Must be 62 or less.")),
        h("div", { class: "field" }, h("label", { text: "Select" }), h("select", { class: "input vk-sel" }, h("option", { text: "qwen3.8:27b-text" }))))),
      card("Chips and pills (icon + word)", h("div", { class: "vk-btnrow" }, chips.map((c) => pill(c[0], c[1], c[2])))));
  },
};

function btnRow(list, ctx) { return h("div", { class: "vk-btnrow" }, btns(list, ctx)); }
function switchBtn(on, onToggle) {
  const b = h("button", { type: "button", role: "switch", class: "vk-switch", "aria-checked": String(on) }, h("span", { class: "track" }, h("i")), h("span", { text: on ? "On" : "Off" }));
  b.addEventListener("click", async () => { on = !on; b.setAttribute("aria-checked", String(on)); b.lastChild.textContent = on ? "On" : "Off"; try { await onToggle(on); } catch (e) { on = !on; b.setAttribute("aria-checked", String(on)); b.lastChild.textContent = on ? "On" : "Off"; toast(errText(e)); } });
  return b;
}

/** Render one section (with its title row). Sections with `poll` refresh themselves in place; vk.later() registers the timer. */
export function renderSection(s, vk = {}) {
  const fn = R[s.k];
  if (!fn) return h("div", { class: "vk-help", text: `Unknown section "${s.k}"` });
  const ctx = { reload: vk.reload, onAct: s.onAct };
  const titled = s.k !== "text" ? (TITLED.has(s.k) && s.title) || (TITLED_IF.has(s.k) && s.title) : !!s.title;
  const make = () => {
    let body;
    try { body = fn(s, ctx); } catch (e) { console.error("viewkit section failed", s.k, e); body = errorBox(e); }
    const head = titled ? h("div", { class: "vk-sech" }, h("h5", { text: s.title }), s.p ? h("span", { class: "vk-ptag", text: "PROPOSAL" }) : null,
      s.k === "table" || s.k === "cards" ? btns(s.acts, { ...ctx, section: s }) : null) : null;
    return h("section", { class: "vk-sec vks-" + s.k }, head, body);
  };
  let el = make();
  if (s.poll && vk.later) {
    const tick = async () => {
      if (!el.isConnected) return false;
      try {
        const patch = await s.poll(s);
        if (patch) { const logBody = el.querySelector(".vk-logbody"), atEnd = !logBody || logBody.scrollHeight - logBody.scrollTop - logBody.clientHeight < 40;
          Object.assign(s, patch); const n = make(); el.replaceWith(n); el = n;
          const nb = el.querySelector(".vk-logbody"); if (nb && atEnd) nb.scrollTop = nb.scrollHeight; }
      } catch (_) { /* keep the last good view; the next tick retries */ }
      return !["done", "failed"].includes(s.st);
    };
    const loop = async () => { if (await tick()) vk.later(loop, s.pollMs || 2000); };
    vk.later(loop, s.pollMs || 2000);
  }
  return el;
}

// ------------------------------------------------------------------------------------------------ the view (tabs + sections)
const lastTab = new Map();

export function renderView(container, view, opts = {}) {
  const root = h("div", { class: "vk" });
  const tabbar = h("div", { class: "vk-tabs", role: "tablist", "aria-label": view.title || "Sections" });
  const body = h("div", { class: "vk-body" });
  root.append(tabbar, body);
  container.replaceChildren(root);
  const tabs = view.tabs || [];
  const id = view.id || view.title || "";
  let cur = null, seq = 0, destroyed = false, tabTimers = [], pollTimer = null;
  const ac = new AbortController();
  const vk = { reload: (o) => show(cur, { silent: true, ...o }), signal: ac.signal,
    later(fn, ms) { const t = setTimeout(fn, ms); tabTimers.push(t); } };
  const stopTab = () => { tabTimers.forEach(clearTimeout); tabTimers = []; clearTimeout(pollTimer); };

  const drawTabs = () => {
    tabbar.hidden = tabs.length < 2;
    tabbar.replaceChildren(...tabs.map((t) => h("button", { type: "button", role: "tab", class: "vk-tab", "aria-selected": String(t.k === cur), id: `vk-tab-${t.k}`,
      onclick: () => show(t.k) }, t.k === cur ? icon("check", 13) : null, t.l)));
  };
  async function show(k, { silent = false } = {}) {
    const t = tabs.find((x) => x.k === k) || tabs.find((x) => x.k === view.tab) || tabs.find((x) => x.k === lastTab.get(id)) || tabs[0];
    if (!t) return;
    const switching = t.k !== cur;
    if (silent && !switching) {                                    // background refresh: never wipe what the user is typing
      const a = document.activeElement;
      if (a && body.contains(a) && /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName)) return schedule(t);
    }
    const my = ++seq;
    stopTab(); cur = t.k; lastTab.set(id, cur); drawTabs();
    let secs;
    if (t.load) {
      if (!silent || switching) body.replaceChildren(loadingBox());
      try { secs = await t.load(vk); }
      catch (e) { if (my === seq && !destroyed) { body.replaceChildren(errorBox(e, () => show(cur))); schedule(t); } return; }
    } else secs = t.s || [];
    if (my !== seq || destroyed) return;
    const nodes = (view.notice ? [view.notice] : []).concat(secs || []).filter(Boolean).map((s) => renderSection(s, vk));
    if (!nodes.length) nodes.push(emptyBox("info", t.emptyTitle || "Nothing here yet", t.empty || ""));
    if (t.needs && t.needs.length && opts.needs !== false) nodes.push(h("div", { class: "vk-needs" }, h("strong", {}, icon("info", 14), "Needs backend"), h("ul", {}, t.needs.map((n) => h("li", { text: n })))));
    body.replaceChildren(...nodes);
    schedule(t);
  }
  function schedule(t) { clearTimeout(pollTimer); if (t.poll && !destroyed) pollTimer = setTimeout(() => show(t.k, { silent: true }), t.poll); }
  const handle = { el: root, tab: () => cur, setTab: (k) => show(k), reload: (o) => show(cur, { silent: true, ...o }),
    destroy() { destroyed = true; stopTab(); ac.abort(); } };
  show(view.tab);
  return handle;
}

/** For a screen's mount(): CSS, header title, cleanup. Returns the handle. */
export function mountView(root, ctx, view, opts) {
  ctx.css("viewkit");
  if (view.title && ctx.shell) ctx.shell.setTitle(view.title);
  const handle = renderView(root, view, opts);
  ctx.onCleanup(handle.destroy);
  return handle;
}
