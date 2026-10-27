// Message composer (design L-Composer): autosizing textarea, "/" command palette, /web toggle, attach, send / stop.
// Modes: "normal" | "streaming" (send becomes Stop) | "offline" | "approval" (disabled, with a note explaining why).
import { h } from "../dom.js";
import { icon } from "../icons.js";

export const COMMANDS = [
  ["/web", "force a web search for the next message"], ["/verify", "check the last answer against fresh sources"], ["/model", "switch model"],
  ["/clear", "wrap up, start fresh"], ["/notes", "session notes"], ["/plan", "show the plan"], ["/project", "set project folder"],
  ["/undo", "undo last file changes"], ["/auto-<dir>", "auto access in a folder"], ["/medium-<dir>", "medium access in a folder"],
];

const draftKey = (id) => "lantern.draft." + (id ?? "new");
const loadDraft = (id) => { try { return localStorage.getItem(draftKey(id)) || ""; } catch (_) { return ""; } };
const saveDraft = (id, v) => { try { v ? localStorage.setItem(draftKey(id), v) : localStorage.removeItem(draftKey(id)); } catch (_) { /* private mode */ } };

export function composer({ sessionId = null, onSend, onStop, onCommand, onAttach, placeholder = "Message Nightage…   / for commands" }) {
  let mode = "normal", web = false, sel = 0, files = [];
  const ta = h("textarea", { rows: 1, placeholder, "aria-label": "Message", autocomplete: "off", enterkeyhint: "send" });
  ta.value = loadDraft(sessionId);
  const webBtn = h("button", { class: "l-web", "aria-pressed": "false", title: "Force a web search for this message", type: "button", onclick: () => toggleWeb() }, icon("globe", 16), "/web");
  const send = h("button", { class: "l-send", type: "button", "aria-label": "Send", onclick: () => submit() }, icon("send", 20));
  const attach = h("button", { class: "l-ib", type: "button", "aria-label": "Attach file", onclick: () => file.click() }, icon("clip", 18));
  const file = h("input", { type: "file", multiple: true, hidden: true });
  const chips = h("div", {});
  const slash = h("div", { class: "l-slash hidden", role: "listbox", "aria-label": "Commands" });
  const note = h("div", { class: "l-composer-note hidden", role: "status" });
  const box = h("div", { class: "l-composer" }, chips, ta, h("div", { class: "row" }, attach, webBtn, h("span", { style: { flex: 1 } }), h("span", { class: "hint only-desktop", text: "Enter sends · Shift+Enter newline" }), send));
  const root = h("div", {}, slash, note, box, file);

  function size() { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 220) + "px"; }
  function toggleWeb(on = !web) { web = on; webBtn.setAttribute("aria-pressed", String(web)); webBtn.lastChild.textContent = "/web" + (web ? " · on" : ""); }
  const base = (c) => c.replace("<dir>", "");
  function matches() {                                    // typing "/au" lists /auto-<dir>; once a path follows, the palette is gone
    const v = ta.value.toLowerCase();
    return v.startsWith("/") && !/\s/.test(v) ? COMMANDS.filter(([c]) => base(c).startsWith(v)) : [];
  }
  function drawSlash() {
    const m = matches();
    slash.classList.toggle("hidden", !m.length || mode !== "normal");
    slash.replaceChildren(...m.map(([c, d], i) => h("div", { role: "option", class: i === sel ? "on" : "", "aria-selected": String(i === sel), onmousedown: (e) => { e.preventDefault(); pick(c); } },
      h("span", { class: "c", text: c }), h("span", { class: "d", text: d }))));
  }
  function pick(c) { ta.value = c.includes("<dir>") ? c.replace("<dir>", "") : c; slash.classList.add("hidden"); ta.focus(); size(); if (!c.includes("<dir>")) submit(); }
  function submit() {
    if (mode === "streaming") return onStop && onStop();
    if (mode !== "normal") return;
    const text = ta.value.trim();
    if (!text && !files.length) return;
    if (text === "/web") { toggleWeb(); ta.value = ""; saveDraft(sessionId, ""); size(); drawSlash(); return; }
    if (text.startsWith("/") && onCommand && onCommand(text)) { ta.value = ""; saveDraft(sessionId, ""); size(); drawSlash(); return; }
    const payload = { web, files: files.slice() };
    ta.value = ""; saveDraft(sessionId, ""); files = []; drawChips(); toggleWeb(false); size(); drawSlash();
    onSend && onSend(text, payload);
  }
  function drawChips() {
    chips.replaceChildren(...files.map((f, i) => h("span", { class: "l-chipfile" }, icon("file", 14), f.name, h("span", { class: "mono", text: Math.max(1, Math.round(f.size / 102.4) / 10) + " kB" }),
      h("button", { class: "l-ib sm", "aria-label": "Remove attachment", onclick: () => { files.splice(i, 1); drawChips(); } }, icon("x", 13)))));
  }

  ta.addEventListener("input", () => { size(); saveDraft(sessionId, ta.value); sel = 0; drawSlash(); });
  ta.addEventListener("keydown", (e) => {
    const m = matches();
    if (!slash.classList.contains("hidden") && m.length) {
      if (e.key === "ArrowDown") { e.preventDefault(); sel = (sel + 1) % m.length; return drawSlash(); }
      if (e.key === "ArrowUp") { e.preventDefault(); sel = (sel - 1 + m.length) % m.length; return drawSlash(); }
      const exact = ta.value.toLowerCase() === base(m[sel][0]);
      if (e.key === "Tab" || (e.key === "Enter" && !exact)) { e.preventDefault(); return pick(m[sel][0]); }
      if (e.key === "Enter" && exact && m[sel][0].includes("<dir>")) { e.preventDefault(); slash.classList.add("hidden"); return; }   // wait for the folder
      if (e.key === "Escape") { slash.classList.add("hidden"); return; }
    }
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
  });
  file.addEventListener("change", () => { files.push(...file.files); file.value = ""; drawChips(); onAttach && onAttach(files); });

  root.setMode = (m, why = "", whyIcon = "lock") => {
    mode = m;
    const off = m === "offline" || m === "approval" || m === "loading";
    box.classList.toggle("off", off);
    ta.disabled = off;
    ta.placeholder = m === "approval" ? "Decide on the action above to continue" : m === "offline" ? "Offline — reconnect to send" : m === "loading" ? "Loading this chat…" : placeholder;
    send.replaceChildren(icon(m === "streaming" ? "stop" : "send", 20));
    send.classList.toggle("stop", m === "streaming");
    send.setAttribute("aria-label", m === "streaming" ? "Stop generating" : "Send");
    send.disabled = off;
    attach.disabled = webBtn.disabled = off;
    note.classList.toggle("hidden", !why);
    note.replaceChildren(...(why ? [icon(whyIcon, 14), why] : []));
    drawSlash();
  };
  root.focus = () => ta.focus();
  root.setValue = (v) => { ta.value = v; size(); saveDraft(sessionId, v); };
  root.getValue = () => ta.value;
  root.forceWeb = () => toggleWeb(true);
  size();
  return root;
}
