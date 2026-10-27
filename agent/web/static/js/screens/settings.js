// Settings (design S-Settings) on the view kit. Every editable setting of agent/settings.py comes from /api/settings and is
// grouped by its own category (one tab each), with type, default, changed/reset and restart-required from the server. Source
// trust rules (/api/trust/rules) get their own tab. Settings that loosen safety ask for a confirmation before the server accepts them.
import { mountView, G, B, T, A, tab } from "../viewkit.js";
import { api } from "../api.js";
import { confirmDialog, toast } from "../ui.js";

const errText = (e) => (e && (e.detail || e.message)) || "Something went wrong.";
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const slug = (c) => c.toLowerCase().replace(/\W+/g, "-").replace(/^-|-$/g, "");
// category (as agent/settings.py names it) -> tab label; an unknown new category still gets a tab of its own
const TAB_LABEL = { Personality: "Personality", Models: "Models", "Memory & chat": "Memory & chat", Performance: "Performance", Coding: "Coding", WhatsApp: "WhatsApp", Safety: "Safety" };
const isPercent = (i) => i.kind === "int" && i.range && i.range[0] === 0 && i.range[1] === 100;
const show = (item, v) => (item.kind === "list" ? (v || []).join("\n") : item.kind === "bool" ? !!v : v ?? "");

/** One viewkit row for a server item. `item` is replaced by every server answer, so changed/default follow the saved state. */
export function settingRow(item, models) {
  const st = { item };
  const k = item.kind;
  const row = { label: item.label, help: item.help + (item.note ? ` ${item.note}` : ""), restart: item.restart_required,
    def: k === "bool" ? (item.default ? "on" : "off") : k === "long_text" ? "(built-in)" : show(item, item.default), val: show(item, item.value) };
  if (k === "bool") row.ctl = "sw";
  else if (k === "choice") { row.ctl = "seg"; row.opts = item.choices; }
  else if (k === "model" && models) {
    const names = [...(item.default === "" ? [{ v: "", l: "(use the background model)" }] : []), ...models.map((n) => ({ v: n, l: n }))];
    if (item.value && !models.includes(item.value)) names.push({ v: item.value, l: `${item.value} (not installed)` });
    row.ctl = "sel"; row.opts = names;
  } else if (k === "long_text" || k === "list") { row.ctl = "ta"; if (k === "list") row.help += " One entry per line."; }
  else if (isPercent(item)) { row.ctl = "sl"; row.min = 0; row.max = 100; row.unit = "%"; }
  else if (k === "int") { row.ctl = "nm"; row.help += item.range ? ` (${item.range[0]} to ${item.range[1]})` : ""; }
  else row.ctl = "tx";
  // viewkit asks `r.changed` and resets it to undefined after a reset; ours always follows the saved value
  Object.defineProperty(row, "changed", { get: () => !same(st.item.value, st.item.default), set() {}, enumerable: true });

  async function send(call) {
    try { return await call(false); }
    catch (e) {
      if (e.status === 409) {                                           // loosens safety: ask, then say so explicitly
        if (!(await confirmDialog(`Loosen safety: ${item.label}?`, errText(e).replace(/^Confirmation needed\.\s*/, ""), { confirm: "Yes, change it", danger: true }))) return null;
        return call(true);
      }
      throw e.status === 422 ? new Error(errText(e).replace(/^invalid value for '[^']*': /, "")) : e;
    }
  }
  row.onChange = async (v) => {
    const value = k === "list" ? String(v).split("\n") : v;
    const r = await send((confirm) => api.put(`/api/settings/${item.key}`, { value, confirm }));
    if (!r) return { val: show(st.item, st.item.value) };               // declined: put the saved value back
    st.item = r;
    return { val: show(r, r.value) };
  };
  row.onReset = async () => {
    const r = await send((confirm) => api.post(`/api/settings/${item.key}/reset?confirm=${confirm}`));
    if (r) st.item = r;
    return { val: show(st.item, st.item.value) };
  };
  return row;
}

const rowsOf = (items, keys, models) => items.filter((i) => keys.includes(i.key)).map((i) => settingRow(i, models));
export { rowsOf };

export async function mount(root, ctx) {
  let cache = null;
  const data = async () => {
    const items = await api.get("/api/settings");
    const models = items.some((i) => i.kind === "model") ? await api.get("/api/models").then((m) => m.available).catch(() => null) : null;
    return { items, models };
  };
  const cats = (items) => { const out = []; for (const i of items) if (!out.includes(i.category)) out.push(i.category); return out; };

  const catTab = (cat) => ({
    k: slug(cat), l: TAB_LABEL[cat] || cat,
    load: async () => {
      cache = await data();
      const own = cache.items.filter((i) => i.category === cat);
      const secs = [];
      if (cat === "Safety") secs.push(B("info", "Your lists extend the built-in ones", "The built-in safe commands and blocked patterns are fixed in the program and cannot be removed here. Anything that loosens safety asks you to confirm first."));
      secs.push(G(cat, own.map((i) => settingRow(i, cache.models))));
      if (cat === "Models") {
        const st = await api.get("/api/status").catch(() => null);
        secs.push(G("GPU mode", [{ label: "GPU mode", help: "Chosen on Models & GPU, where changing it also restarts the model server.", ctl: "tx", val: st ? st.gpu_mode || "not chosen yet" : "unknown", def: "", disabled: true }]),
          A([{ l: "Open Models & GPU", icon: "cpu", onClick: () => ctx.navigate("#/models") }]));
      }
      if (cat === "Memory & chat") secs.push(G("Reasoning", [{ label: "Show reasoning", help: "Not available yet: the model's thoughts are never shown.", ctl: "sw", val: false, def: "off", disabled: true }]));
      if (cat === "Safety") {
        const acc = await api.get("/api/access").catch(() => null);
        const n = acc ? new Set([...acc.greenlist, ...acc.session.map((s) => s.path)]).size : null;
        secs.push(G("Commands and folders", [
          { label: "Dangerous actions", help: "Fixed: dangerous commands always ask first and blocked ones never run. There is no setting to change this.", ctl: "seg:Always ask|Never run", val: "Always ask", def: "Always ask", disabled: true },
          { label: "Allowed folders", help: "Managed on Projects & folder access.", ctl: "tx", val: n == null ? "unknown" : `${n} folder${n === 1 ? "" : "s"}`, def: "", disabled: true }]),
          A([{ l: "Open Projects & folder access", icon: "folder", onClick: () => ctx.navigate("#/projects") }]));
      }
      if (cat === "WhatsApp") secs.push(G("Sending", [{ label: "Sending", help: "Blocked by policy: the product never sends WhatsApp messages for you.", ctl: "sw", val: false, def: "off", disabled: true }]));
      return secs;
    },
  });

  const trustTab = {
    k: "trust", l: "Source trust",
    load: async (vk) => {
      const rules = await api.get("/api/trust/rules");
      const rows = [
        ...rules.allow.map((d) => [d, "ok:allow (high trust)", d]), ...rules.deny.map((d) => [d, "block:deny (hidden)", d]),
        ...Object.entries(rules.tiers).map(([d, t]) => [d, `info:pinned ${t}`, d]),
      ].map(([d, tag, dom]) => ({ cells: [{ t: d, mono: true }, tag, { t: "Remove", btn: true, onClick: async () => { await api.put("/api/trust/rules", { domain: dom, action: "clear" }); toast("Rule removed"); vk.reload(); } }] }));
      const draft = { domain: "", rule: "allow" };
      const send = async (c) => {
        const [action, tier] = draft.rule.startsWith("T") ? ["tier", draft.rule] : [draft.rule, undefined];
        try { await api.put("/api/trust/rules", { domain: draft.domain, action, tier }); } catch (e) { throw new Error(errText(e)); }
        toast("Rule saved"); c.reload();
      };
      return [
        T("Your domain rules", ["Domain", "Rule", ""], rows, []),
        G("Add a rule", [
          { label: "Domain", help: "For example example.org. Sub-domains follow their main domain.", ctl: "tx", val: "", def: "", onChange: (v) => { draft.domain = v; } },
          { label: "Rule", help: "Allow raises a domain to high trust, deny hides it from answers, a tier pins its rating (T1 best, T4 worst).",
            ctl: "sel", opts: [{ v: "allow", l: "Allow" }, { v: "deny", l: "Deny" }, { v: "T1", l: "Pin tier T1 (best)" }, { v: "T2", l: "Pin tier T2" }, { v: "T3", l: "Pin tier T3" }, { v: "T4", l: "Pin tier T4 (worst)" }], val: "allow", def: "allow", onChange: (v) => { draft.rule = v; } }]),
        A([{ l: "Save rule", primary: true, onClick: send }]),
        G("Not available yet", [
          { label: "Trust sensitivity", help: "Not available yet: pages are rated automatically; only your domain rules above change the rating.", ctl: "seg:Lenient|Balanced|Strict", val: "Balanced", def: "Balanced", disabled: true },
          { label: "Show passage in source cards", help: "Not available yet: source cards always show their passage.", ctl: "sw", val: true, def: "on", disabled: true }]),
      ];
    },
  };

  let first;
  try { first = await data(); } catch (e) { first = null; }
  const catTabs = first ? cats(first.items).map(catTab)
    : [{ k: "all", l: "Settings", load: async () => { await data(); throw new Error("The settings are reachable again: open Settings once more."); } }];
  mountView(root, ctx, { title: "Settings", id: "settings", tabs: [...catTabs, trustTab] });
}
