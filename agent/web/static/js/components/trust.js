// Trust chip: icon + word (+ first reason), never colour alone. Levels from the server: high | medium | low | avoid | unknown.
import { h } from "../dom.js";
import { icon } from "../icons.js";

const M = {
  high: ["shield-check", "high", "ok", ""], medium: ["shield-half", "medium", "info", ""], low: ["shield-minus", "low", "warn", ""],
  avoid: ["shield-x", "avoid", "danger", "thick"], unknown: ["shield-q", "unknown", "neutral", "dashed"],
  allowed: ["user", "allowed by you", "ok", "dashed"], blocked: ["blocked", "blocked by you", "block", "thick"],
};

/** The level to show for a `trust` object from a sources event (user rules win over the computed level). */
export function levelOf(trust) {
  if (!trust) return "unknown";
  if (trust.user_rule === "allow") return "allowed";
  if (trust.user_rule === "deny") return "blocked";
  return trust.level || "unknown";
}

export function trustChip({ level = "unknown", reason = "", compact = false } = {}) {
  const [ic, word, v, style] = M[level] || M.unknown;
  return h("span", { class: `l-trust v-${v} ${style} ${compact ? "compact" : ""}`.trim(), title: `Trust: ${word}${reason ? " — " + reason : ""}` },
    icon(ic, 14), h("b", { text: word }), reason ? h("span", { class: "why", text: "· " + reason }) : null);
}
