// Small pieces shared by the Projects, Code agent, Audit and Status screens: state boxes, status pills, meters, copyable commands.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { copyText } from "../ui.js";

/** Icon + word on a semantic background (never colour alone). tone: ok | info | warn | danger | block | neutral */
export const pill = (tone, iconName, word, cls = "") => h("span", { class: `l-tag v-${tone} ${cls}`.trim() }, icon(iconName, 13), word);

export const loadingBox = (text = "Loading…") => h("div", { class: "l-state", role: "status", "aria-live": "polite" }, h("div", { class: "l-skel", style: { height: "14px", width: "60%", borderRadius: "var(--radius-md)" } }), h("span", { class: "mono", text }));

export const emptyBox = (iconName, title, text, ...extra) => h("div", { class: "l-state center" }, icon(iconName, 26), h("strong", { text: title }), text ? h("span", { style: { color: "var(--color-text-2)", maxWidth: "46ch" }, text }) : null, ...extra);

/** A failed load: what happened (the server's wording) and a retry button. */
export function errorBox(err, retry) {
  const offline = err && err.status === 0;
  return h("div", { class: "l-state v-danger", role: "alert" },
    h("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, icon(offline ? "wifioff" : "danger", 18), h("strong", { text: offline ? "Cannot reach the server" : "Could not load this" })),
    h("span", { text: (err && (err.detail || err.message)) || "Unknown error" }),
    retry ? h("button", { class: "btn btn-secondary", style: { minHeight: "44px" }, onclick: retry }, icon("refresh", 15), "Try again") : null);
}

/** A thin meter. frac 0..1; tone picks the fill. The numbers next to it carry the meaning, the bar is decoration. */
export const meter = (frac, tone = "accent") => h("div", { class: "ops-meter", "aria-hidden": "true" },
  h("i", { style: { width: `${Math.round(Math.max(0, Math.min(1, frac || 0)) * 100)}%`, background: tone === "accent" ? "var(--color-accent)" : `var(--${tone})` } }));

/** A shell command to run in a terminal, with a copy button. */
export const commandBox = (cmd, label = "Copy command") => h("div", { class: "ops-cmd" },
  h("code", { text: cmd }),
  h("button", { class: "btn btn-secondary", style: { minHeight: "44px" }, "aria-label": label, onclick: () => copyText(cmd, "Command copied") }, icon("copy", 15), "Copy"));

export const sectionHead = (text) => h("h5", { class: "l-section-h", text });

/** 1280 -> "1.3 GB" style from MB. */
export const gb = (mb) => (mb == null ? "—" : (mb / 1024).toFixed(1));
export const dash = (v, unit = "") => (v == null || v === "" ? "—" : `${v}${unit}`);
