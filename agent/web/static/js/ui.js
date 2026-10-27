// Small shared UI helpers: toast, confirm dialog, clipboard, file download.
import { h } from "./dom.js";

let toastTimer = null;
export function toast(text, ms = 2600) {
  document.querySelectorAll(".l-toast").forEach((n) => n.remove());
  const t = h("div", { class: "l-toast", role: "status", text });
  document.body.append(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), ms);
}

export async function copyText(text, done = "Copied") {
  try { await navigator.clipboard.writeText(text); toast(done); }
  catch (_) {                                   // clipboard API needs https or localhost: fall back to a hidden textarea
    const ta = h("textarea", { style: { position: "fixed", opacity: 0 } });
    ta.value = text; document.body.append(ta); ta.select();
    try { document.execCommand("copy"); toast(done); } catch (e) { toast("Could not copy"); }
    ta.remove();
  }
}

export function download(filename, content, type = "text/plain") {
  const url = URL.createObjectURL(content instanceof Blob ? content : new Blob([content], { type }));
  const a = h("a", { href: url, download: filename });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Modal dialog. Resolves with the id of the pressed action (or null on Esc / backdrop). */
export function dialog({ title, body, actions = [{ id: "ok", label: "OK", primary: true }] }) {
  return new Promise((resolve) => {
    const close = (v) => { backdrop.remove(); document.removeEventListener("keydown", onKey); resolve(v); };
    const onKey = (e) => { if (e.key === "Escape") close(null); };
    const box = h("div", { class: "l-dialog", role: "dialog", "aria-modal": "true", "aria-label": title },
      h("h4", { text: title }), typeof body === "string" ? h("p", { text: body }) : body,
      h("div", { class: "btns" }, actions.map((a) => h("button", { class: `btn ${a.danger ? "danger" : a.primary ? "btn-primary" : "btn-secondary"}`, onclick: () => close(a.id) }, a.label))));
    const backdrop = h("div", { class: "l-dialog-backdrop", onclick: (e) => { if (e.target === backdrop) close(null); } }, box);
    document.addEventListener("keydown", onKey);
    document.body.append(backdrop);
    (box.querySelector(".btn-primary, .danger") || box.querySelector("button"))?.focus();
  });
}

export const confirmDialog = (title, body, { confirm = "Confirm", danger = false } = {}) =>
  dialog({ title, body, actions: [{ id: "no", label: "Cancel" }, { id: "yes", label: confirm, primary: !danger, danger }] }).then((r) => r === "yes");
