// Audit log (design S-Audit): a read-only chronology of every tool and shell action, with the outcome in words.
import { h, fmtDateTime, debounce } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { pill, loadingBox, errorBox, emptyBox } from "../components/ops-ui.js";

const PAGE = 100, MAX = 500, CLIP = 200;
const OUTCOME = { approved: ["check", "ok", "approved"], denied: ["x", "warn", "denied"], blocked: ["blocked", "block", "blocked"], error: ["danger", "danger", "error"] };

/** Tool rows store {"name", "arguments"} as JSON; show it as `name key=value ...`. Shell rows are the command itself. */
function what(row) {
  if (row.kind !== "tool") return row.detail || "";
  try {
    const d = JSON.parse(row.detail);
    if (d && typeof d.name === "string") {
      const args = Object.entries(d.arguments || {}).map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join(" ");
      return args ? `${d.name} ${args}` : d.name;
    }
  } catch (_) { /* plain text */ }
  return row.detail || "";
}

function outcome(row) {
  const r = String(row.result || "");
  if (row.approved) return "approved";
  if (r.startsWith("BLOCKED")) return "blocked";
  if (r.startsWith("Error")) return "error";
  return "denied";
}

export async function mount(root, ctx) {
  ctx.css("ops-ui"); ctx.css("s-audit");
  let kind = "", q = "", limit = PAGE, seq = 0;

  const search = h("input", { class: "input", type: "search", placeholder: "Search actions", "aria-label": "Search actions", autocomplete: "off",
    oninput: debounce(() => { q = search.value.trim(); limit = q ? MAX : PAGE; load(); }, 300) });
  const seg = h("div", { class: "seg", role: "radiogroup", "aria-label": "Kind" },
    [["", "All"], ["tool", "Tool"], ["shell", "Shell"]].map(([v, l]) => h("label", { class: "seg-opt" },
      h("input", { type: "radio", name: "s-audit-kind", value: v, checked: v === kind, onchange: () => { kind = v; limit = q ? MAX : PAGE; load(); } }), l)));
  const body = h("div", {});
  root.replaceChildren(h("div", { class: "l-page" },
    h("div", { class: "s-audit-bar" }, h("div", { class: "s-audit-search" }, search, icon("search", 16)), seg),
    h("div", { class: "s-audit-ro" }, icon("lock", 13), "Read-only. Every tool and shell action is recorded here."),
    body));

  const outcomeTag = (row) => { const [ic, tone, word] = OUTCOME[outcome(row)]; return pill(tone, ic, word, outcome(row) === "blocked" ? "s-audit-block" : ""); };

  function result(text) {
    const t = String(text || "");
    if (t.length <= CLIP) return h("span", { class: "s-audit-res", text: t });
    const span = h("span", { class: "s-audit-res", text: t.slice(0, CLIP) + "…" });
    let open = false;
    return h("span", {}, span, " ", h("button", { class: "btn btn-ghost", style: { minHeight: "32px" }, "aria-expanded": "false", onclick: (e) => {
      open = !open; span.textContent = open ? t : t.slice(0, CLIP) + "…"; e.currentTarget.textContent = open ? "less" : "more"; e.currentTarget.setAttribute("aria-expanded", String(open));
    } }, "more"));
  }

  function table(rows) {
    return h("table", { class: "table s-audit-table" },
      h("thead", {}, h("tr", {}, ["Time", "Kind", "What", "Outcome", "Result"].map((t) => h("th", { scope: "col", text: t })))),
      h("tbody", {}, rows.map((r) => h("tr", {},
        h("td", { class: "s-audit-time", text: fmtDateTime(r.ts) }), h("td", {}, h("span", { class: "mono", text: r.kind })),
        h("td", { class: "s-audit-what", text: what(r) }), h("td", {}, outcomeTag(r)), h("td", {}, result(r.result))))));
  }

  const cards = (rows) => h("div", { class: "s-audit-cards" }, rows.map((r) => h("div", { class: "s-audit-card" },
    h("div", { class: "s-audit-card-top" }, h("span", { class: "s-audit-time", text: `${fmtDateTime(r.ts)} · ${r.kind}` }), outcomeTag(r)),
    h("span", { class: "s-audit-what", text: what(r) }), result(r.result))));

  async function load() {
    const mine = ++seq;
    body.replaceChildren(loadingBox("Loading the log…"));
    let rows;
    try { rows = await api.get(`/api/audit?limit=${limit}${kind ? `&kind=${kind}` : ""}${q ? `&q=${encodeURIComponent(q)}` : ""}`); }
    catch (e) { if (mine === seq) body.replaceChildren(errorBox(e, load)); return; }
    if (mine !== seq) return;                                   // a newer search/filter is already on its way
    if (!rows.length) {
      body.replaceChildren(q || kind
        ? emptyBox("search", "Nothing matches", "No recorded action fits this search. Clear it to see everything.")
        : emptyBox("list", "Nothing recorded yet", "Tool and shell actions the assistant takes will appear here, newest first."));
      return;
    }
    const more = !q && rows.length >= limit && limit < MAX
      ? h("button", { class: "btn btn-secondary s-audit-more", style: { minHeight: "44px", marginTop: "var(--space-3)" }, onclick: () => { limit = Math.min(MAX, limit + PAGE); load(); } }, "Show older")
      : null;
    body.replaceChildren(h("div", { class: "s-audit-wide" }, table(rows)), h("div", { class: "s-audit-narrow" }, cards(rows)), more,
      !q && rows.length >= MAX ? h("p", { class: "mono", text: `Showing the newest ${MAX} actions.` }) : null);
  }

  await load();
}
