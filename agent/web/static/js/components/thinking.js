// Working indicator (design L-Thinking): rotating lantern rays, a sheen over the status word, a seconds counter,
// skeleton lines while thinking, or a typed code pane when `code` lines are given (coding / editing).
// thinking({ kind, detail, seconds, code }) -> element with .set(kind, detail) and .stop(). The timer stops by itself when
// the element leaves the document. Reduced motion is handled globally in tokens.css (animations switch off, the seconds still count).
import { h, svg } from "../dom.js";

export const THINK_LABEL = { thinking: "Thinking", planning: "Planning", searching: "Searching the web", reading: "Reading pages", coding: "Writing code",
  editing: "Editing", verifying: "Running checks", compacting: "Compacting context" };
/** Example code for demos (the design's typed sample). Real screens pass their own `code`. */
export const DEMO_CODE = "def play(rounds):\n    score = Counter()\n    for _ in range(rounds):\n        move = ask_move()\n        outcome = judge(move, pick())\n        score[outcome] += 1\n    return score\n";

const fm = (s) => (s < 60 ? s + " s" : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"));

export function thinking({ kind = "thinking", detail = "", mood = "", seconds = 0, code = null, file = "" } = {}) {
  const rays = svg("svg", { width: 26, height: 26, viewBox: "-13 -13 26 26", overflow: "visible", class: "spin" },
    ...[0, 1, 2, 3, 4, 5, 6, 7].map((i) => svg("g", { transform: `rotate(${i * 45})` },
      svg("line", { x1: 0, y1: -4.5, x2: 0, y2: -10.5, stroke: "currentColor", "stroke-width": 2.2, "stroke-linecap": "round", style: `animation:l-fade 1.6s ease-in-out ${i * 0.2}s infinite` }))));
  // `mood` is the optional emotes side-channel (agent/mood.py): a tiny kaomoji next to the
  // label, e.g. "O_o" while searching - never required, never changes what `kind`/`detail` mean.
  const label = h("span", { class: "label" }), det = h("span", { class: "detail" }), moodEl = h("span", { class: "mood" }), secs = h("span", { class: "secs" });
  const body = h("div", { class: "body" });
  const el = h("div", { class: "l-think", role: "status", "aria-live": "polite" },
    h("div", { class: "row" }, h("span", { class: "glow", "aria-hidden": "true" }, rays), label, moodEl, det,
      h("span", { class: "meta" }, h("span", { class: "dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")), secs)), body);
  let sec = seconds, typed = 0, k = kind, m = mood, t1 = null, t2 = null;
  const lines = code ? String(code).split("\n") : null;
  const draw = () => {
    label.textContent = THINK_LABEL[k] || k || "Working";
    det.textContent = detail; det.hidden = !detail;
    moodEl.textContent = m || ""; moodEl.hidden = !m;
    secs.textContent = fm(sec);
    const showCode = lines && (k === "coding" || k === "editing");
    if (showCode) {
      const full = lines.join("\n"), n = Math.min(typed, full.length), cur = full.slice(0, n).split("\n");
      body.replaceChildren(h("div", { class: "code" },
        h("div", { class: "ch" }, h("span", { text: file || detail || "" }), h("span", { text: `${cur.length} lines · writing` })),
        h("div", { class: "cl" }, cur.map((t, i) => h("div", { class: i === cur.length - 1 ? "ln last" : "ln" }, h("span", { class: "no", text: i + 1 }), h("span", { class: "tx" }, t, i === cur.length - 1 ? h("span", { class: "caret" }) : null))))));
    } else if (["thinking", "planning", "compacting"].includes(k)) {
      body.replaceChildren(h("div", { class: "bars", "aria-hidden": "true" }, h("i", { style: { width: "86%" } }), h("i", { style: { width: "64%", animationDelay: ".3s" } }), h("i", { style: { width: "42%", animationDelay: ".6s" } })));
    } else body.replaceChildren();
  };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced && lines) typed = lines.join("\n").length;
  draw();
  const stop = () => { clearInterval(t1); clearInterval(t2); };
  t1 = setInterval(() => { if (!el.isConnected) return stop(); sec += 1; secs.textContent = fm(sec); }, 1000);
  if (lines && !reduced) t2 = setInterval(() => {
    if (!el.isConnected) return stop();
    const total = lines.join("\n").length;
    typed = typed > total + 26 ? 0 : typed + 2;
    if (k === "coding" || k === "editing") draw();
  }, 55);
  el.set = (nk, nd = "", nm = "") => { k = nk; detail = nd; m = nm; draw(); };
  el.stop = stop;
  return el;
}
