// The mark (lantern artwork, to be redrawn separately) and the "Nightage" wordmark.
import { h, svg } from "./dom.js";

export function mark(size = 28, word = true) {
  const s = svg("svg", { width: size, height: size, viewBox: "0 0 32 32", fill: "none", "aria-hidden": "true" },
    svg("path", { d: "M13 5.5c0-1.4 1.1-2.5 3-2.5s3 1.1 3 2.5", stroke: "var(--color-accent)", "stroke-width": 1.6, "stroke-linecap": "round", transform: "translate(-0.8 -0.5)", opacity: 0.9 }),
    svg("ellipse", { cx: 17, cy: 19.8, rx: 3.4, ry: 5.2, fill: "var(--color-accent-2)", opacity: 0.5, style: "mix-blend-mode:var(--plate-blend)" }),
    svg("ellipse", { cx: 16, cy: 19.2, rx: 3.4, ry: 5.2, fill: "var(--color-flame)" }),
    svg("path", { d: "M12.5 5.5h7M10 8.5h12l1.5 3v11L21 26H11l-2.5-3.5v-11z", stroke: "currentColor", "stroke-width": 1.7, "stroke-linejoin": "round" }),
    svg("path", { d: "M12 26v2.5h8V26", stroke: "currentColor", "stroke-width": 1.7, "stroke-linejoin": "round" }));
  return h("span", { class: "l-mark" }, s, word ? h("span", { class: "word" }, "Nightage") : null);
}
