// Prompt studio (design S-Prompt): a rough prompt in, an improved or anonymised one out. Both calls run on the local
// model and are slow, so the output pane shows progress (elapsed time) and an error with retry.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { copyText } from "../ui.js";
import { renderMarkdown } from "../md.js";

const MAX = 20000;
const STRENGTHS = [
  ["light", "Light", "Remove only names and obvious personal details; keep the domain."],
  ["heavy", "Medium", "Strip personal details and reframe as a neutral task (recommended)."],
  ["poison", "Heavy", "As Medium, plus generic cover so who and why cannot be inferred."],
  ["max", "Max", "Five passes: distil the core, then rebuild it unrecognisably. Slowest."],
];
const errText = (e) => (e && (e.detail || e.message)) || "Something went wrong.";

export async function mount(root, ctx) {
  ctx.css("s-prompt");
  ctx.shell.setTitle("Prompt studio");

  let tab = "improved", strength = "heavy", run = null, timer = null, token = 0;
  const results = { improved: null, anon: null };           // { text, source, note? }
  const pane = { improved: null, anon: null };              // "busy" state per tab: { t0 } | { error } | null

  const raw = h("textarea", { id: "s-prompt-raw", class: "input s-prompt-raw", maxlength: MAX + 1000, placeholder: "Paste or write a rough prompt…", oninput: () => { updateInput(); } });
  const count = h("span", { class: "s-prompt-count mono" });
  const hint = h("span", { class: "s-prompt-hint", id: "s-prompt-hint" });
  const improveBtn = h("button", { class: "btn btn-primary s-prompt-hit", onclick: () => go("improved") }, icon("sparkle", 15), "Improve");
  const anonBtn = h("button", { class: "btn btn-secondary s-prompt-hit", onclick: () => go("anon") }, icon("shield-half", 15), "Anonymise");
  const seg = h("div", { class: "seg s-prompt-seg", role: "radiogroup", "aria-label": "Anonymise strength" }, STRENGTHS.map(([v, l, d]) =>
    h("label", { class: "seg-opt s-prompt-hit", title: d }, h("input", { type: "radio", name: "s-prompt-st", value: v, checked: v === strength || null, onchange: () => { strength = v; hint.textContent = d; } }), l)));
  hint.textContent = STRENGTHS[1][2];

  const tabs = [["improved", "Improved"], ["anon", "Anonymised"]].map(([k, l]) =>
    h("button", { type: "button", role: "tab", id: "s-prompt-tab-" + k, "aria-controls": "s-prompt-out", class: "s-prompt-tab s-prompt-hit", onclick: () => { tab = k; drawOut(); }, onkeydown: (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { tab = k === "improved" ? "anon" : "improved"; drawOut(); document.getElementById("s-prompt-tab-" + tab).focus(); } } }, l));
  const copyBtn = h("button", { class: "btn btn-secondary s-prompt-hit s-prompt-copy", onclick: () => { const r = results[tab]; if (r) copyText(r.text, "Copied the prompt"); } }, icon("copy", 15), "Copy");
  const out = h("div", { id: "s-prompt-out", class: "s-prompt-out", role: "tabpanel", "aria-live": "polite" });
  const saved = h("div", { class: "s-prompt-saved" });

  root.replaceChildren(h("div", { class: "l-page s-prompt" },
    h("div", { class: "s-prompt-cols" },
      h("div", { class: "s-prompt-col" },
        h("label", { for: "s-prompt-raw", class: "l-section-h", text: "Your text" }), raw,
        h("div", { class: "s-prompt-bar" }, count, h("span", { class: "s-prompt-label", text: "Strength" }), seg),
        hint, h("div", { class: "s-prompt-bar" }, anonBtn, improveBtn)),
      h("div", { class: "s-prompt-col" },
        h("div", { class: "s-prompt-bar" }, h("div", { class: "s-prompt-tabs", role: "tablist", "aria-label": "Result" }, tabs), copyBtn), out)),
    saved));

  function updateInput() {
    const n = raw.value.trim().length;
    count.textContent = `${n.toLocaleString("en")} / ${MAX.toLocaleString("en")}`;
    count.classList.toggle("over", n > MAX);
    const dis = !n || n > MAX || !!run;
    improveBtn.disabled = anonBtn.disabled = dis;
    if (results[tab] && results[tab].source !== raw.value.trim() && results[tab].source !== "saved") drawOut();
  }

  async function go(kind) {
    const text = raw.value.trim();
    if (!text || text.length > MAX || run) return;
    const mine = ++token;
    tab = kind; run = { kind, t0: Date.now() }; pane[kind] = { t0: Date.now() };
    updateInput(); drawOut();
    clearInterval(timer);
    timer = setInterval(() => { const s = out.querySelector(".s-prompt-secs"); if (s) s.textContent = Math.round((Date.now() - run.t0) / 1000) + " s"; }, 1000);
    try {
      const r = kind === "improved" ? await api.post("/api/prompt/improve", { text }) : await api.post("/api/prompt/anonymize", { text, strength });
      if (mine !== token) return;
      results[kind] = { text: r.text || "", source: text, note: kind === "anon" ? STRENGTHS.find((s) => s[0] === strength)[1] : "" };
      pane[kind] = null;
    } catch (e) { if (mine !== token) return; pane[kind] = { error: errText(e), retry: kind }; }
    run = null; clearInterval(timer); updateInput(); drawOut();
  }
  const stop = () => { token++; if (run) pane[run.kind] = null; run = null; clearInterval(timer); updateInput(); drawOut(); };
  ctx.onCleanup(() => { token++; clearInterval(timer); });

  let drawId = 0;
  async function drawOut() {
    const mineDraw = ++drawId;
    for (const t of tabs) { const sel = t.id === "s-prompt-tab-" + tab; t.setAttribute("aria-selected", String(sel)); t.tabIndex = sel ? 0 : -1; }
    out.setAttribute("aria-labelledby", "s-prompt-tab-" + tab);
    const p = pane[tab], r = results[tab];
    copyBtn.disabled = !r || !!p;
    if (p && p.error) {
      out.replaceChildren(h("div", { class: "l-state", role: "alert" }, icon("danger", 18), h("strong", { text: "That did not work." }), h("span", { text: p.error }),
        h("button", { class: "btn btn-secondary s-prompt-hit", onclick: () => go(p.retry) }, "Try again")));
    } else if (p) {
      out.replaceChildren(h("div", { class: "s-prompt-busy", role: "status" },
        h("div", { class: "s-prompt-busyhead" }, h("strong", { text: tab === "anon" && strength === "max" ? "Anonymising in five passes…" : tab === "anon" ? "Anonymising…" : "Improving…" }), h("span", { class: "mono s-prompt-secs", text: "0 s" })),
        h("div", { class: "s-prompt-bar2", role: "progressbar", "aria-valuetext": "in progress" }, h("i")),
        h("span", { class: "s-prompt-hint", text: "Your own model is writing this; it can take a minute. Nothing leaves this machine." }),
        h("button", { class: "btn btn-ghost s-prompt-hit", onclick: stop }, "Stop waiting")));
    } else if (r) {
      const stale = r.source !== "saved" && r.source !== raw.value.trim();
      const body = r.text.trim() ? await renderMarkdown(r.text) : null;      // sanitised by md.js
      if (mineDraw !== drawId) return;                                       // a newer draw won while rendering
      out.replaceChildren(...[
        r.source === "saved" ? h("span", { class: "s-prompt-hint", text: "From a saved prompt chat." }) : null,
        stale ? h("span", { class: "s-prompt-hint warn" }, icon("warn", 13), "Your text changed since this was made.") : null,
        body || h("span", { class: "s-prompt-hint", text: "The model returned nothing. Try again." }),
      ]);
    } else {
      out.replaceChildren(h("div", { class: "s-prompt-empty" }, icon(tab === "anon" ? "shield-half" : "sparkle", 22),
        h("strong", { text: tab === "anon" ? "Nothing anonymised yet." : "Nothing improved yet." }),
        h("span", { text: tab === "anon" ? "Write your text, pick a strength and press Anonymise. It removes names and details that point to you." : "Write a rough prompt on the left and press Improve. You get a sharper version to paste anywhere." })));
    }
  }

  async function loadSaved() {
    let rows;
    try { rows = await api.get("/api/prompt/sessions"); } catch (_) { saved.replaceChildren(); return; }     // optional extra: stay quiet
    saved.replaceChildren(h("span", { text: "Saved prompt chats:" }), ...(rows.length ? rows.slice(0, 8).map((s) =>
      h("button", { class: "s-prompt-chip", title: s.mode, onclick: () => { if (!s.last) return; const k = s.mode === "anon" ? "anon" : "improved"; results[k] = { text: s.last, source: "saved" }; pane[k] = null; tab = k; drawOut(); } },
        `${s.title || "Untitled"} · ${s.created}`)) : [h("span", { class: "s-prompt-none", text: "none yet" })]));
  }

  updateInput(); drawOut();
  await loadSaved();
}
