// Tools & audit: the tool registry with risk tiers, the audit log with filters and CSV export, and the (not web-startable) audit packs.
import { api } from "../api.js";
import { download } from "../ui.js";
import { mountView, B, C, F, T } from "../viewkit.js";

const RISK = { SAFE: "ok:SAFE", CAUTION: "warn:CAUTION", DANGEROUS: "danger:DANGEROUS", BLOCKED: "block:BLOCKED", DYNAMIC: "info:BY COMMAND" };
const DECISION = { approved: "ok:approved", denied: "warn:denied", refused: "block:refused", error: "danger:error" };
const time = (ts) => (ts ? ts.replace("T", " ").slice(5, 16) : "—");
const clip = (s, n = 80) => { const t = String(s || "").replace(/\s+/g, " "); return t.length > n ? t.slice(0, n - 1) + "…" : t; };
const csv = (rows) => rows.map((r) => r.map((c) => `"${String(c ?? "").replace(/"/g, '""')}"`).join(",")).join("\n");
const argsOf = (r) => { try { const d = JSON.parse(r.detail); return d && d.name ? JSON.stringify(d.arguments || {}) : r.detail; } catch (_) { return r.detail; } };

export async function mount(root, ctx) {
  let q = "", decision = "";

  async function toolsLoad() {
    const tools = await api.get("/api/tools");
    return [T("Every tool the agent can call", ["Tool", "Description", "Risk", "Enabled", "Last use"],
      tools.map((t) => [`m:${t.name}`, clip(t.description, 90), RISK[t.risk] || t.risk, t.risk === "BLOCKED" ? "block:blocked" : "ok:on", time(t.last_use)])),
      B("info", "No per-tool switch", "Every registered tool can be called; the risk tier decides whether it runs, needs your approval, or is refused.")];
  }

  async function auditLoad(vk) {
    const rows = await api.get(`/api/tools/audit?limit=300&q=${encodeURIComponent(q)}&decision=${decision}`);
    return [
      F([{ l: "Search", v: q, i: "search", input: true, placeholder: "tool, argument", onInput: (v) => { q = v.trim(); vk.reload(); } },
         { l: "Decision", v: decision, i: "check", options: [{ v: "", l: "any" }, "approved", "denied", "refused", "error"], onInput: (v) => { decision = v; vk.reload(); } }]),
      { k: "table", title: "Actions", cols: ["Time", "Tool", "Arguments", "Decision", "Result"],
        rows: rows.map((r) => [`m:${time(r.ts)}`, `m:${r.tool || r.kind}`, `m:${clip(argsOf(r), 60)}`, DECISION[r.decision], clip(r.result, 50)]),
        acts: [{ l: "Export CSV", onClick: () => download("audit-log.csv", csv([["time", "kind", "tool", "arguments", "decision", "result"], ...rows.map((r) => [r.ts, r.kind, r.tool, argsOf(r), r.decision, r.result])]), "text/csv") }] },
      { k: "ld", title: "Detail", items: rows.slice(0, 50).map((r) => ({ t: `${time(r.ts)} · ${r.tool || r.kind}`, sub: r.decision, tag: DECISION[r.decision], _r: r })),
        detail: null, detailFor: (it) => ({ title: `${it._r.tool || it._r.kind}`, rows: [["Arguments", argsOf(it._r)], ["Decision", it._r.decision], ["Result", it._r.result || "—"]], acts: [] }), emptyText: "Nothing recorded yet." },
    ];
  }

  async function packsLoad() {
    const p = await api.get("/api/tools/packs");
    return [C("Optional packs", p.packs.map((x) => ({ t: x.name, s: x.summary }))), B("warn", "Not started from the web GUI", p.reason)];
  }

  mountView(root, ctx, { title: "Tools & audit", tabs: [
    { k: "tools", l: "Tools", load: toolsLoad }, { k: "audit", l: "Audit log", load: auditLoad }, { k: "packs", l: "Tool packs", load: packsLoad } ] });
}
