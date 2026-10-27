// One source card (design L-Source): mark, title, domain · date, trust chip, status/flags, highlighted passage, feedback.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { trustChip, levelOf } from "./trust.js";

const STATUS = { cached: ["clock", "cached", "neutral"], skipped: ["file", "skipped", "warn"], failed: ["x", "failed", "warn"], timeout: ["clock", "timeout", "warn"], blocked: ["lock", "blocked", "warn"] };
const UNREADABLE = {
  paywall: "Unreadable: behind a paywall. Only the headline was seen.", "cookie wall": "Unreadable: cookie wall.", "needs JavaScript": "Unreadable: needs JavaScript.",
  captcha: "Unreadable: captcha.", pdf: "Skipped: PDF, not opened to save time.", "http 403": "Unreadable: server refused (403).",
};

const tag = (v, ...kids) => h("span", { class: `l-tag v-${v}` }, ...kids);
const safeHref = (u) => (/^https?:\/\//i.test(u || "") ? u : null);

export function sourceCard(item, { compact = false, active = false, onFeedback = null } = {}) {
  const st = item.status || "ok", flags = item.flags || (item.trust && item.trust.flags) || [];
  const suspicious = flags.includes("suspicious_instructions");
  const unread = ["failed", "timeout", "blocked", "skipped"].includes(st);
  const dim = suspicious || st === "failed" || st === "timeout";
  const p = item.passage || "";
  const cut = p.search(/[,.;]/), k = cut > 20 ? cut + 1 : Math.min(p.length, 50);
  const domain = (item.domain || "").replace(/^www\./, "");
  const href = safeHref(item.url);
  const [sIcon, sText, sVar] = STATUS[st] || ["info", "", "neutral"];

  return h("article", { class: `l-source ${dim ? "dim" : ""} ${active ? "active" : ""}`.trim(), dataset: { n: item.n } },
    h("div", { class: "top" },
      h("span", { class: "mark", "aria-hidden": "true", text: domain.slice(0, 2).toUpperCase() }),
      h("div", { style: { minWidth: 0, flex: 1 } },
        h("div", { class: "ttl" }, h("span", { class: "n", text: `[${item.n}]` }),
          href ? h("a", { href, target: "_blank", rel: "noopener noreferrer", style: { color: "inherit", textDecoration: "none" }, text: item.title || domain }) : (item.title || domain)),
        h("div", { class: "dom", text: `${domain} · ${item.date || "undated"}` }))),
    h("div", { class: "tags" },
      trustChip({ level: levelOf(item.trust), reason: (item.trust && item.trust.reasons && item.trust.reasons[0]) || "", compact: true }),
      st !== "ok" ? tag(sVar, icon(sIcon, 13), sText + (item.status_note ? " · " + item.status_note : "")) : null,
      item.duplicate_of ? tag("neutral", icon("link", 13), `same text as [${item.duplicate_of}] · counts once`) : null,
      flags.filter((f) => !["suspicious_instructions"].includes(f)).map((f) => tag(f === "typosquat" ? "danger" : "warn", icon("flag", 12), f.replace(/_/g, " ")))),
    suspicious ? h("p", { class: "bad" }, h("strong", { text: "Ignored:" }), " this page tried to give the assistant instructions. Its text is not used.") : null,
    p && !unread && !suspicious ? h("p", {}, h("mark", { text: p.slice(0, k) }), p.slice(k)) : null,
    unread ? h("p", { class: "muted", text: UNREADABLE[item.status_note] || (st === "timeout" ? "Timed out. Nothing was read from this page." : "Could not be read.") }) : null,
    !compact && !dim && st !== "skipped" ? h("div", { class: "acts" },
      h("button", { class: "l-ghost", onclick: () => onFeedback && onFeedback("allow", item) }, icon("up", 14), "Good source"),
      h("button", { class: "l-ghost", onclick: () => onFeedback && onFeedback("tier-low", item) }, icon("down", 14), "Poor"),
      h("button", { class: "l-ghost", onclick: () => onFeedback && onFeedback("deny", item) }, icon("blocked", 14), "Block domain")) : null);
}
