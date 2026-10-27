// The evidence inspector (design L-Sources): verdict, Sources / Conflict tabs, source cards, "Verify this answer".
// Desktop: right-hand panel. Phone: bottom sheet (the shell positions it; `mode` only adds the grabber).
import { h, fmtDateTime } from "../dom.js";
import { icon } from "../icons.js";
import { sourceCard } from "./source.js";
import { verdictBox } from "./verdict.js";
import { trustChip } from "./trust.js";

export function sourcesPanel({ sources, verdict, view = "sources", mode = "panel", onClose, onVerify, onFeedback, onRules, activeN = null }) {
  const items = (sources && sources.items) || [];
  const conflicts = (verdict && verdict.conflicts) || [];
  let current = view === "conflict" && !conflicts.length ? "sources" : view;
  const body = h("div", { class: "body" });
  const tabs = h("div", { class: "tabs", role: "tablist" });

  const draw = () => {
    tabs.replaceChildren(
      h("button", { role: "tab", "aria-selected": String(current === "sources"), onclick: () => { current = "sources"; draw(); } }, `Sources · ${items.length}`),
      conflicts.length ? h("button", { role: "tab", "aria-selected": String(current === "conflict"), onclick: () => { current = "conflict"; draw(); } }, icon("conflict", 14), `Conflict · ${conflicts.length}`) : null);
    body.replaceChildren();
    if (current === "conflict") {
      for (const c of conflicts) {
        body.append(h("p", { class: "l-note" }, "Who says what about ", h("strong", { text: c.attribute })));
        c.values.forEach((v, i) => {
          const pref = c.preferred !== undefined ? String(v.value) === String(c.preferred) : i === 0;
          body.append(h("div", { class: `l-conflict-row ${pref ? "pref" : ""}`.trim() },
            h("div", {}, h("span", { class: "val", text: String(v.value) }), pref ? h("span", { class: "pr" }, icon("check", 14), "preferred") : null),
            h("div", { class: "meta" }, h("span", { class: "mono", text: v.sources.map((n) => `[${n}]`).join(" ") }), trustChip({ level: v.trust, compact: true }))));
        });
        if (c.why) body.append(h("p", { class: "l-note" }, h("strong", { text: "Why:" }), " " + c.why));
      }
    } else {
      for (const it of items) body.append(sourceCard(it, { active: activeN === it.n, onFeedback }));
      if (!items.length) body.append(h("div", { class: "l-state" }, "No sources were read for this answer."));
    }
  };
  draw();

  return h("div", { class: "l-sources" },
    mode === "sheet" ? h("div", { class: "grab", "aria-hidden": "true" }) : null,
    h("div", { class: "hd" }, h("h4", { text: "Evidence" }),
      h("span", { class: "mono", text: [sources && sources.engine, sources && sources.searched_at && fmtDateTime(sources.searched_at)].filter(Boolean).join(" · ") }),
      h("button", { class: "l-ib sm", "aria-label": "Close evidence", onclick: onClose }, icon("x", 18))),
    verdict ? h("div", { style: { padding: "0 var(--space-3)" } }, verdictBox(verdict)) : null,
    tabs, body,
    h("div", { class: "foot" },
      h("button", { class: "btn btn-primary", onclick: onVerify }, icon("refresh", 16), "Verify this answer"),
      h("button", { class: "btn btn-secondary", onclick: onRules }, "Trust rules")));
}
