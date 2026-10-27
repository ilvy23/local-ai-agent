// Evidence strip: one line with the verdict (icon + word + summary). Six states from the `verdict` event.
import { h } from "../dom.js";
import { icon } from "../icons.js";

const M = {
  verified: ["verified", "Verified", "ok", false], likely: ["likely", "Likely", "info", false], single_source: ["single", "Single source", "warn", false],
  conflicting: ["conflict", "Sources conflict", "danger", false], low_trust: ["lowtrust", "Low-trust / unverified", "danger", true], from_memory: ["memory", "From memory", "neutral", true],
};
const DEFAULT_SUMMARY = { from_memory: "no sources — not checked against the web" };

export function verdictBox(ev, { compact = false, onDetails = null } = {}) {
  const [ic, word, v, dashed] = M[ev.label] || M.from_memory;
  const summary = ev.summary || DEFAULT_SUMMARY[ev.label] || "";
  return h("div", { class: `l-verdict v-${v} ${dashed ? "dashed" : ""} ${compact ? "compact" : ""}`.trim(), role: "status" },
    icon(ic, 18), h("b", { text: word }), !compact && summary ? h("span", { class: "sum", text: summary }) : null,
    !compact && onDetails && ev.label !== "from_memory" ? h("button", { class: "more", onclick: onDetails }, "details", icon("chevr", 14)) : null);
}
