// Knowledge: semantic memory search, index maintenance, the skill curator, notes pointers - bound to the memory routes.
import { api } from "../api.js";
import { download, toast } from "../ui.js";
import { mountView, B, F, S, T, A, X } from "../viewkit.js";

export async function mount(root, ctx) {
  let query = "";
  const ago = (iso) => { if (!iso) return "never"; const d = Math.round((Date.now() - new Date(iso).getTime()) / 86400000); return d <= 0 ? "today" : `${d} d ago`; };

  async function searchLoad(vk) {
    const secs = [F([{ l: "Semantic search", v: query, i: "search", input: true, placeholder: "what did I say about GPUs", onInput: (v) => { query = v.trim(); vk.reload(); } }])];
    if (!query) return [...secs, B("info", "Type to search", "Searches the facts you and the assistant stored, ranked by meaning.")];
    const r = await api.get("/api/memory/search?q=" + encodeURIComponent(query));
    if (r.mode === "substring") secs.push(B("warn", "Plain text match", "The embedding index could not be used, so this is a substring search without similarity scores."));
    secs.push(T(`Results (${r.results.length})`, ["Fact", "Similarity", "Kind"],
      r.results.map((x) => [x.content, x.similarity == null ? "—" : "m:" + x.similarity.toFixed(2), "neutral:" + x.kind])));
    return secs;
  }

  async function maintainLoad(vk) {
    const s = await api.get("/api/memory/status");
    const un = Object.entries(s.unindexed || {}).filter(([, n]) => n).map(([k, n]) => `${n} ${k}(s)`).join(", ");
    const secs = [
      S([["Embedding model", s.embed_model, "local"], ["Index", s.healthy ? "healthy" : s.available ? "needs repair" : "unavailable", s.healthy ? "ok" : un],
         ["Vectors", String(s.vectors), s.dim ? `${s.dim}-dim` : ""]]),
      { k: "job", title: "Re-embed memory", sub: "Rebuilds the search index with the configured embedding model", pct: 0, st: "queued", el: "—", eta: "—", log: [],
        acts: [{ l: "Start", primary: true, onClick: async () => { await api.post("/api/memory/reembed"); vk.reload(); } }] },
      A([{ l: "Repair index", icon: "sliders", onClick: async () => { const r = await api.post("/api/memory/repair"); toast("Repaired: " + JSON.stringify(r.repaired)); vk.reload(); } },
         { l: "Export memory", icon: "download", onClick: async () => { download("memory-export.json", JSON.stringify(await api.get("/api/memory/export"), null, 2), "application/json"); } },
         { l: "Import memory…", icon: "download", disabled: true }]),
      B("info", "Import isn't available yet", "The product can export facts, but has no import path; add facts one by one in Memory."),
    ];
    const st = await api.get("/api/memory/reembed/status");
    if (st.state !== "idle") {
      const pct = st.total ? Math.round(st.done / st.total * 100) : 0;
      secs[1] = { ...secs[1], pct, st: st.state === "running" ? "running" : st.state === "done" ? "done" : "failed", el: `${st.done}/${st.total}`, eta: st.eta_seconds ? `${Math.ceil(st.eta_seconds / 60)} min left` : "—", log: st.error ? ["E: " + st.error] : [],
        poll: async () => { const x = await api.get("/api/memory/reembed/status"); return { pct: x.total ? Math.round(x.done / x.total * 100) : 0, st: x.state === "running" ? "running" : x.state === "done" ? "done" : "failed", el: `${x.done}/${x.total}`, eta: x.eta_seconds ? `${Math.ceil(x.eta_seconds / 60)} min left` : "—" }; }, pollMs: 2000,
        acts: [] };
    }
    return secs;
  }

  async function curatorLoad(vk) {
    const r = await api.get("/api/skills-curator/report");
    const rows = r.archived.map((a) => [a.title, "neutral:archived (" + a.reason + ")"]);
    return [
      S([["Would archive", String(r.would.archived), "now"], ["Would merge", String(r.would.merged), ""], ["Keep", String(r.would.kept), `last run ${ago(r.last_ran)}`]]),
      T("Archived by the curator", ["Skill", "Finding"], rows, [{ l: "Run curator now", primary: true, onClick: async () => { const x = await api.post("/api/skills-curator/run"); toast(`Archived ${x.archived}, merged ${x.merged}, kept ${x.kept}`); vk.reload(); } }]),
      X("Thresholds", `Archive at failure ratio ${r.settings.max_failure_ratio} or unused for ${r.settings.max_age_days} days (config: curator).`),
      B("info", "No schedule yet", "Nothing runs the curator automatically; run it from here."),
    ];
  }

  mountView(root, ctx, { title: "Knowledge", tabs: [
    { k: "search", l: "Memory search", load: searchLoad },
    { k: "maintain", l: "Maintenance", load: maintainLoad },
    { k: "curator", l: "Skill curator", load: curatorLoad },
    { k: "notes", l: "Notes", s: [
      B("info", "Notes live with their chat or project", "Session notes: Sessions > Notes & plan. Project notes: the same tab and the Projects screen. There is no separate global scratch note in the product."),    ] },
  ] });
}
