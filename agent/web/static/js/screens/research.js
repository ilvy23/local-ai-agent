// Research, trust & web: domain rules, feedback history, search engines/connectors/cache, verify history - bound to /api/trust and /api/research.
import { h } from "../dom.js";
import { api } from "../api.js";
import { dialog, download, toast } from "../ui.js";
import { mountView, B, C, F, G, LD, R, S, T } from "../viewkit.js";

const tierName = { allow: "ok:allow", deny: "block:deny" };
const mb = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " kB");
const hours = (s) => (s >= 86400 ? s / 86400 + " d" : s >= 3600 ? s / 3600 + " h" : s / 60 + " min");

export async function mount(root, ctx) {
  let filter = "";

  async function rulesLoad(vk) {
    const doc = await api.get("/api/trust/rules");
    const rows = [
      ...doc.allow.map((d) => [d, "allow", "High trust"]),
      ...doc.deny.map((d) => [d, "deny", "Hidden from answers"]),
      ...Object.entries(doc.tiers).map(([d, t]) => [d, "tier " + t, "Medium or lower, as you set"]),
    ].filter((r) => r[0].includes(filter.toLowerCase()));
    const remove = (d) => async () => { await api.put("/api/trust/rules", { domain: d, action: "clear" }); vk.reload(); };
    const add = async () => {
      const dom = h("input", { class: "input", placeholder: "example.org", "aria-label": "Domain" });
      const act = h("select", { class: "input", "aria-label": "Rule" }, ["allow", "deny", "T1", "T2", "T3", "T4"].map((x) => h("option", { value: x, text: x })));
      if ((await dialog({ title: "Add domain rule", body: h("div", { class: "l-list" }, dom, act), actions: [{ id: "x", label: "Cancel" }, { id: "ok", label: "Add", primary: true }] })) !== "ok") return;
      const v = act.value;
      await api.put("/api/trust/rules", v === "allow" || v === "deny" ? { domain: dom.value, action: v } : { domain: dom.value, action: "tier", tier: v });
      vk.reload();
    };
    const importRules = () => new Promise((res) => {
      const f = h("input", { type: "file", accept: ".json,application/json" });
      f.onchange = async () => { try { await api.post("/api/research/rules/import", JSON.parse(await f.files[0].text())); toast("Rules imported"); vk.reload(); } catch (e) { toast(e.detail || "Could not import"); } res(); };
      f.click();
    });
    const explain = async (item) => {
      const x = await api.get("/api/research/explain?domain=" + encodeURIComponent(item._d));
      return { title: item._d, rows: [["Rating", `${x.tier} · ${x.level}`], ["Reasons", x.reasons.join(" · ") || "none"], ["Your rule", x.user_rule || "none"]],
               acts: [{ l: "Remove my rule", onClick: remove(item._d) }] };
    };
    return [
      F([{ l: "Search domains", v: filter, i: "search", input: true, placeholder: "rust-lang.org", onInput: (v) => { filter = v.trim(); vk.reload(); } }]),
      { k: "table", title: "Your domain rules", cols: ["Domain", "Rule", "Effect", ""],
        rows: rows.map(([d, r, e]) => [`m:${d}`, tierName[r] || "info:" + r, e, { t: "Remove", btn: true, onClick: remove(d) }]),
        acts: [{ l: "Add domain", primary: true, onClick: add }, { l: "Import", onClick: importRules }, { l: "Export", onClick: () => download("source-trust.json", JSON.stringify(doc, null, 2), "application/json") }] },
      { k: "ld", title: "Why this rating", items: rows.map(([d, r]) => ({ t: d, sub: r, tag: tierName[r] || "info:" + r, _d: d })), detailFor: explain, emptyText: "Add a rule to look at it here." },
    ];
  }

  async function feedbackLoad() {
    const f = await api.get("/api/research/feedback");
    const tone = { good: "ok:good", poor: "warn:poor", block: "block:blocked" };
    return [
      T("Per-source feedback", ["Domain", "You said", "When"], f.rows.map((r) => [`m:${r.domain}`, tone[r.said] || "info:" + r.said,
        f.rules_changed_at ? "rules last changed " + new Date(f.rules_changed_at * 1000).toLocaleDateString() : "—"])),
      B("info", "No per-click history", "Feedback from the chat is saved as a rule, so this is the current state of those rules; the product keeps no timestamp for each click."),
    ];
  }

  async function webLoad(vk) {
    const [engines, conns, cache, ssrf] = await Promise.all([api.get("/api/research/engines"), api.get("/api/research/connectors"), api.get("/api/research/cache"), api.get("/api/research/ssrf")]);
    const rows = engines.map((e) => [e.label, e.active ? "ok:in the chain" : "neutral:off (" + e.how + ")",
      e.active ? { t: "Test", btn: true, onClick: async () => { const r = await api.post(`/api/research/engines/${e.name}/test`); toast(`${e.label}: ${r.status} (${r.detail})`, 4200); } } : ""]);
    return [
      T("Engine chain", ["Engine", "Status", ""], rows),
      B("info", "Engines are set by environment variables", "AGENT_SEARXNG_URL and BRAVE_API_KEY are read when the server starts; there is no key field here on purpose."),
      C("Connectors", conns.map((c) => ({ t: c.name, s: "used when: " + c.when }))),
      B("info", "Connectors have no switch", "Each one answers automatically when a question matches its pattern; the product has no per-connector off switch."),
      G("Cache", [
        { label: "Cache freshness", help: "Fixed in the product", ctl: "tx", val: `news ${hours(cache.ttl_seconds.news)} · other ${hours(cache.ttl_seconds.default)} · docs ${hours(cache.ttl_seconds.docs)}`, def: "", disabled: true },
        { label: "Cache size", help: `${cache.pages} pages, ${cache.queries} searches`, ctl: "tx", val: mb(cache.bytes), def: "", disabled: true },
      ]),
      { k: "actions", items: [{ l: "Clear web cache", onClick: async () => { await api.del("/api/memory/web-cache"); toast("Cache cleared"); vk.reload(); } }] },
      B("ok", "SSRF guard on", "Local, private and Tailscale addresses are never fetched. Read-only. Checked: " + ssrf.samples.map((s) => `${s.url} ${s.blocked ? "blocked" : "allowed"}`).join("; ")),
    ];
  }

  async function verifyLoad() {
    const runs = await api.get("/api/research/verify-history");
    const tone = { supported: "ok:", contradicted: "danger:", unclear: "warn:" };
    const tag = (v) => (v ? (tone[v] || "info:") + v : "neutral:none");
    return [T("“Verify this answer” runs", ["Answer", "Before", "After"], runs.map((r) => [r.answer || r.chat, tag(r.before), tag(r.after)])),
      ...(runs.length ? [] : [B("info", "No verify runs yet", "Press Verify this answer under a web answer in the chat.")])];
  }

  mountView(root, ctx, { title: "Research, trust & web", tabs: [
    { k: "rules", l: "Source trust", load: rulesLoad }, { k: "feedback", l: "Feedback history", load: feedbackLoad },
    { k: "web", l: "Web search", load: webLoad }, { k: "verify", l: "Verify history", load: verifyLoad } ] });
}
