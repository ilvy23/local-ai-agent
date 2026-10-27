// Visual answer blocks (design L-Block). renderBlock(block) -> <figure>. Blocks come from the `present` tool as
// {id, kind, title, spec, source_ids, provenance}. Everything from the server goes in via textContent / attributes;
// SVG illustrations are shown through <img> (scripts cannot run there) and html widgets in a sandboxed iframe.
import { h, svg, clear, append } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, copyText, download, dialog } from "../ui.js";
import { ensureMermaid, ensureHljs } from "../vendor.js";
import { effectiveTheme } from "../store.js";

const KIND = {
  chart: ["chart", "chart"], table: ["list", "table"], diagram: ["code", "diagram"], compare: ["sliders", "comparison"], timeline: ["clock", "timeline"],
  stats: ["chart", "stat tiles"], callout: ["info", "callout"], quote: ["book", "quote"], steps: ["list", "steps"], code: ["code", "code"], diff: ["code", "diff"],
  math: ["sparkle", "formula"], map: ["pin", "map"], gallery: ["image", "gallery"], html: ["bolt", "widget"], svg: ["image", "illustration"],
};
// series styles differ in dash AND marker shape, so no chart relies on colour alone
const SERIES = [
  { color: "var(--series-1)", dash: "", marker: "diamond", glyph: "◆" }, { color: "var(--series-2)", dash: "8 4", marker: "square", glyph: "■" },
  { color: "var(--series-3)", dash: "2 4", marker: "triangle", glyph: "▲" }, { color: "var(--series-4)", dash: "12 4 2 4", marker: "circle", glyph: "●" },
  { color: "var(--series-5)", dash: "", marker: "x", glyph: "✕" }, { color: "var(--series-6)", dash: "4 3", marker: "plus", glyph: "✚" },
];
const style = (i) => ({ ...SERIES[i % SERIES.length], color: i >= SERIES.length ? `color-mix(in srgb, var(--color-text) ${40 + (i % 3) * 15}%, var(--color-bg))` : SERIES[i % SERIES.length].color });
const num = (v) => (typeof v === "number" ? v : parseFloat(String(v).replace(/[^\d.+\-eE]/g, "")));
const fmt = (v) => (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 1 }) : String(Math.round(v * 100) / 100));
const fmtX = (v) => (Number.isInteger(v) && v >= 1000 && v <= 3000 ? String(v) : fmt(v));
const txt = (x, y, t, attrs = {}) => svg("text", { x, y, "font-family": "var(--font-mono)", "font-size": 11, fill: "var(--color-text-3)", ...attrs }, t);

// ------------------------------------------------------------------------------------------------ chart

function niceScale(min, max) {
  if (max === min) max = min + 1;
  const raw = (max - min) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw))), step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw);
  return [Math.floor(min / step) * step, Math.ceil(max / step) * step];
}

function markerShape(kind, x, y, color, filled = true) {
  const common = filled ? { fill: color } : { fill: "var(--color-bg)", stroke: color, "stroke-width": 2 };
  if (kind === "x") return svg("path", { d: `M${x - 4} ${y - 4}L${x + 4} ${y + 4}M${x + 4} ${y - 4}L${x - 4} ${y + 4}`, fill: "none", stroke: color, "stroke-width": 2.2 });
  if (kind === "plus") return svg("path", { d: `M${x - 5} ${y}H${x + 5}M${x} ${y - 5}V${y + 5}`, fill: "none", stroke: color, "stroke-width": 2.2 });
  if (kind === "square") return svg("rect", { x: x - 4, y: y - 4, width: 8, height: 8, ...common });
  if (kind === "triangle") return svg("path", { d: `M${x} ${y - 5}L${x + 5} ${y + 4}L${x - 5} ${y + 4}z`, ...common });
  if (kind === "diamond") return svg("path", { d: `M${x} ${y - 5}L${x + 5} ${y}L${x} ${y + 5}L${x - 5} ${y}z`, ...common });
  return svg("circle", { cx: x, cy: y, r: 3.5, ...common });
}

function chartSummary(spec) {
  return `${spec.type.replace("_", " ")} chart: ` + spec.series.map((s) => {
    const p = s.points; return `${s.name} from ${fmt(num(p[0][p[0].length - 1]))} to ${fmt(num(p[p.length - 1][p[p.length - 1].length - 1]))}`;
  }).join("; ");
}

function legend(series) {
  return h("div", { class: "l-legend" }, series.map((s, i) => {
    const st = style(i);
    return h("span", {}, svg("svg", { width: 22, height: 8, "aria-hidden": "true" }, svg("path", { d: "M0 4h22", stroke: st.color, "stroke-width": 2.4, "stroke-dasharray": st.dash })), `${st.glyph} ${s.name}`);
  }));
}

function chartBlock(spec) {
  const type = spec.type, series = spec.series;
  const W = 560, H = 290, L = 56, R = 14, T = 30, B = 46, iw = W - L - R, ih = H - T - B;
  const root = h("div", {});
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": chartSummary(spec), class: "l-chart" });

  if (["heatmap", "radar"].includes(type)) {                       // not drawn yet: show the numbers instead of nothing
    root.append(h("p", { class: "l-note", style: { marginBottom: "8px" } }, icon("info", 14), `A ${type} is not drawn yet; here are the numbers.`));
    root.append(tableBlock({ columns: [{ key: "x", label: "" }, ...series.map((x, i) => ({ key: "s" + i, label: x.name, type: "number" }))],
      rows: series[0].points.map((p, j) => ({ x: p[0], ...Object.fromEntries(series.map((x, i) => ["s" + i, x.points[j] ? x.points[j][x.points[j].length - 1] : ""])) })) }));
    return root;
  }

  if (type === "pie" || type === "donut") {
    const pts = series[0].points.map((p) => [String(p[0]), num(p[p.length - 1])]).filter((p) => p[1] > 0), total = pts.reduce((a, p) => a + p[1], 0) || 1;
    const cx = W / 2, cy = H / 2, r = 105, inner = type === "donut" ? 58 : 0;
    const fills = SERIES.map((x) => x.color).concat("var(--color-neutral-500)");
    let a0 = -Math.PI / 2;
    const lg = h("div", { class: "l-legend" });
    pts.forEach(([label, v], i) => {
      const a1 = a0 + (v / total) * Math.PI * 2, large = a1 - a0 > Math.PI ? 1 : 0, pt = (rad, a) => `${cx + rad * Math.cos(a)},${cy + rad * Math.sin(a)}`;
      const d = inner ? `M${pt(r, a0)} A${r},${r} 0 ${large} 1 ${pt(r, a1)} L${pt(inner, a1)} A${inner},${inner} 0 ${large} 0 ${pt(inner, a0)}Z` : `M${cx},${cy} L${pt(r, a0)} A${r},${r} 0 ${large} 1 ${pt(r, a1)}Z`;
      s.append(svg("path", { d, fill: fills[i % fills.length], stroke: "var(--color-surface)", "stroke-width": 1.5 }));
      lg.append(h("span", {}, h("i", { style: { display: "inline-block", width: "10px", height: "10px", background: fills[i % fills.length] } }), `${label} ${fmt(v)} (${Math.round((v / total) * 100)}%)`));
      a0 = a1;
    });
    root.append(s, lg);
    return root;
  }

  if (type === "bar" && series.length === 1 && series[0].points.length <= 14) {   // the design's horizontal bar list
    const pts = series[0].points.map((p) => [String(p[0]), num(p[p.length - 1])]);
    const max = Math.max(...pts.map((p) => p[1]), 1), sqrt = max / Math.max(Math.min(...pts.map((p) => p[1] || max)), 1e-9) > 20;
    const unit = (spec.y && spec.y.unit) || "";
    const list = h("div", { class: "l-bars", role: "img", "aria-label": chartSummary(spec) });
    pts.forEach(([label, v], i) => list.append(h("div", { class: "r" }, h("span", { text: label }),
      h("span", { class: "t" }, h("i", { style: { width: Math.max(3, (sqrt ? Math.sqrt(v / max) : v / max) * 100) + "%", background: `var(--series-${(i % 6) + 1})`, opacity: 1 } })),
      h("span", { class: "v", text: `${fmt(v)}${unit ? " " + unit : ""}` }))));
    append(root, [list, sqrt ? h("div", { class: "cap" }, "bars on a square-root scale (values differ a lot)") : null]);
    return root;
  }

  const cats = [...new Set(series.flatMap((se) => se.points.map((p) => String(p[0]))))];
  const numericX = type === "scatter" || (!["bar", "stacked_bar", "histogram"].includes(type) && cats.every((c) => !isNaN(parseFloat(c))));
  const ys = series.flatMap((se) => se.points.map((p) => num(p[p.length - 1])));
  const stacked = type === "stacked_bar" || spec.stacked;
  const valAt = (se, c) => num((se.points.find((p) => String(p[0]) === c) || [0, 0]).slice(-1)[0]);
  let ymax = stacked ? Math.max(...cats.map((c) => series.reduce((a, se) => a + valAt(se, c), 0))) : Math.max(...ys, 0), ymin = Math.min(0, ...ys);
  if (spec.y && typeof spec.y.min === "number") ymin = spec.y.min; else [ymin] = niceScale(ymin, ymax);
  if (spec.y && typeof spec.y.max === "number") ymax = spec.y.max; else [, ymax] = niceScale(ymin, ymax);
  const ypx = (v) => T + ih - ((v - ymin) / (ymax - ymin || 1)) * ih;
  for (let i = 0; i <= 4; i++) {
    const v = ymin + ((ymax - ymin) * i) / 4;
    s.append(svg("line", { x1: L, x2: W - R, y1: ypx(v), y2: ypx(v), stroke: "var(--color-divider)", "stroke-dasharray": i ? "2 4" : "" }));
    s.append(txt(L - 6, ypx(v) + 4, fmt(v), { "text-anchor": "end" }));
  }
  let xpx;
  if (numericX) {
    const xs = series.flatMap((se) => se.points.map((p) => num(p[0]))), x0 = Math.min(...xs), x1 = Math.max(...xs);
    xpx = (x) => L + (x1 === x0 ? iw / 2 : ((num(x) - x0) / (x1 - x0)) * iw);
    const ticks = x1 === x0 ? [x0] : [x0, (x0 + x1) / 2, x1];
    ticks.forEach((x, i) => s.append(txt(xpx(x), H - B + 18, fmtX(x), { "text-anchor": i === 0 ? "start" : i === ticks.length - 1 ? "end" : "middle" })));
  } else {
    const step = iw / cats.length;
    xpx = (c) => L + step * (cats.indexOf(String(c)) + 0.5);
    cats.forEach((c, i) => { if (cats.length <= 14 || i % Math.ceil(cats.length / 14) === 0) s.append(txt(xpx(c), H - B + 18, c.slice(0, 12), { "text-anchor": "middle" })); });
  }
  if (spec.x && spec.x.label) s.append(txt(L + iw / 2, H - 6, spec.x.label, { "text-anchor": "middle", "font-family": "var(--font-body)", "font-size": 12 }));
  if (spec.y && (spec.y.label || spec.y.unit)) s.append(txt(4, 14, [spec.y.label, spec.y.unit && `(${spec.y.unit})`].filter(Boolean).join(" "), { "font-family": "var(--font-body)", "font-size": 12 }));

  series.forEach((se, si) => {
    const st = style(si), pts = se.points.map((p) => [p[0], num(p[p.length - 1])]);
    if (["bar", "stacked_bar", "histogram"].includes(type)) {
      const step = iw / cats.length, bw = stacked ? step * 0.7 : (step * 0.8) / series.length;
      pts.forEach(([c, v]) => {
        const base = stacked ? series.slice(0, si).reduce((a, o) => a + valAt(o, String(c)), 0) : 0;
        const x = L + step * cats.indexOf(String(c)) + (stacked ? step * 0.15 : step * 0.1 + bw * si), y1 = ypx(base + v), y0 = ypx(Math.max(base, ymin));
        s.append(svg("rect", { x, y: Math.min(y0, y1), width: Math.max(bw, 1), height: Math.abs(y0 - y1), fill: st.color, opacity: si ? 0.8 : 1 }));
      });
    } else if (type === "scatter") {
      pts.forEach(([x, v]) => s.append(markerShape(st.marker, xpx(x), ypx(v), st.color)));
    } else {
      const d = pts.map(([x, v], i) => `${i ? "L" : "M"}${xpx(x)},${ypx(v)}`).join(" ");
      if (type === "area") s.append(svg("path", { d: `${d} L${xpx(pts[pts.length - 1][0])},${ypx(Math.max(0, ymin))} L${xpx(pts[0][0])},${ypx(Math.max(0, ymin))}Z`, fill: st.color, opacity: 0.18 }));
      s.append(svg("path", { d, fill: "none", stroke: st.color, "stroke-width": 2.4, "stroke-dasharray": st.dash }));
      pts.forEach(([x, v]) => s.append(markerShape(st.marker, xpx(x), ypx(v), st.color, si % 2 === 0)));
    }
  });
  for (const a of spec.annotations || []) {
    if (a.x === undefined) continue;
    s.append(svg("path", { d: `M${xpx(a.x)} ${T}v${ih}`, stroke: "var(--color-text-2)", "stroke-dasharray": "3 3" }));
    s.append(svg("text", { x: xpx(a.x) + 4, y: T + 12, "font-size": 12, fill: "var(--color-text-2)", "font-family": "var(--font-body)", "font-style": "italic" }, a.text || ""));
  }
  append(root, [s, series.length > 1 ? legend(series) : null]);
  return root;
}

// ------------------------------------------------------------------------------------------------ the other kinds

function tableBlock(spec) {
  const cols = spec.columns, rows = spec.rows.slice();
  let sort = spec.sort ? { key: spec.sort.key, dir: spec.sort.dir === "desc" ? -1 : 1 } : null;
  const body = h("tbody"), head = h("tr");
  const best = {};
  if (spec.highlight === "best") for (const c of cols) if (c.type === "number" && c.better) {
    const vals = rows.map((r) => r[c.key]).filter((v) => typeof v === "number");
    if (vals.length) best[c.key] = c.better === "lower" ? Math.min(...vals) : Math.max(...vals);
  }
  const draw = () => {
    clear(body);
    const data = rows.slice();
    if (sort) data.sort((a, b) => { const x = a[sort.key], y = b[sort.key]; return (typeof x === "number" && typeof y === "number" ? x - y : String(x ?? "").localeCompare(String(y ?? ""))) * sort.dir; });
    for (const r of data) {
      const isBest = Object.keys(best).some((k) => r[k] === best[k]);
      body.append(h("tr", { class: isBest ? "best" : "" }, cols.map((c, i) => h("td", { class: typeof r[c.key] === "number" ? "num" : "" },
        r[c.key] == null ? "" : String(r[c.key]), isBest && i === 0 ? h("span", { class: "bestmark", text: "✓ best" }) : null))));
    }
  };
  for (const c of cols) head.append(h("th", { class: c.type === "number" ? "num" : "", "aria-sort": "none" },
    h("button", { title: "Sort", onclick: () => { sort = { key: c.key, dir: sort && sort.key === c.key ? -sort.dir : 1 }; draw(); } },
      c.label + (c.unit ? ` (${c.unit})` : ""), icon("chevu", 12))));
  draw();
  const wrap = h("div", { style: { overflowX: "auto" } }, h("table", {}, h("thead", {}, head), body),
    h("div", { class: "cap" }, [spec.caption, `${rows.length} row${rows.length === 1 ? "" : "s"}`].filter(Boolean).join(" · ")));
  wrap.csv = () => [cols.map((c) => c.label), ...rows.map((r) => cols.map((c) => r[c.key] ?? ""))].map((l) => l.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  return wrap;
}

async function drawMermaid(host, source) {
  try {
    const m = await ensureMermaid(effectiveTheme() === "dark");
    await m.parse(source);
    const { svg: out } = await m.render("m" + Math.random().toString(36).slice(2), source);
    host.innerHTML = out;                                 // securityLevel "strict": mermaid sanitises its own SVG
    host.dataset.svg = out;
    return true;
  } catch (_) { return false; }
}

function diagramBlock(spec) {
  const root = h("div", {});
  const holder = h("div", { style: { overflowX: "auto" }, role: "img", "aria-label": "Diagram (mermaid " + (spec.type || "flowchart") + ")" });
  const src = h("pre", { class: "raw hidden", text: spec.source });
  const note = h("div", { class: "l-note hidden", style: { marginTop: "6px" } }, icon("warn", 14), "The diagram could not be drawn; here is its source.");
  holder.append(h("div", { class: "l-skeleton" }, h("i"), h("i"), h("i")));
  drawMermaid(holder, spec.source).then((ok) => {
    if (!ok) { holder.remove(); src.classList.remove("hidden"); note.classList.remove("hidden"); }
  });
  root.append(holder, note, src, h("div", { class: "cap", style: { display: "flex", gap: "12px", flexWrap: "wrap" } },
    h("span", { text: `${spec.type || "diagram"} · mermaid source ${spec.source.length} B` }),
    h("button", { class: "l-link", style: { marginLeft: "auto" }, onclick: () => src.classList.toggle("hidden") }, "show source")));
  root.svg = () => holder.dataset.svg || "";
  return root;
}

function compareBlock(spec) {
  const n = spec.options.length, cols = `90px repeat(${n}, 1fr)`;
  const grid = h("div", { class: "l-compare", style: { gridTemplateColumns: cols } }, h("span"));
  spec.options.forEach((o, j) => grid.append(h("strong", { class: j === spec.winner ? "win" : "" }, String(o), j === spec.winner ? h("span", { style: { fontSize: "11px" }, text: "★ winner" }) : null)));
  const bar = (v) => `${fmt(v)} ` + "▮".repeat(Math.max(0, Math.min(10, Math.round(v)))) + "▯".repeat(Math.max(0, 10 - Math.round(v)));
  spec.criteria.forEach((c, i) => {
    grid.append(h("span", { text: `${c.name} ×${c.weight}` }));
    spec.scores[i].forEach((v) => grid.append(h("span", { class: "mono", style: { color: "var(--color-text)", fontSize: "13px" }, text: bar(v) })));
  });
  const wsum = spec.criteria.reduce((a, c) => a + c.weight, 0) || 1;
  grid.append(h("strong", { text: "Weighted" }));
  spec.totals.forEach((t, j) => grid.append(h("strong", { class: "mono", style: { color: j === spec.winner ? "var(--ok)" : "var(--color-text)" }, text: fmt(t / wsum) })));
  (["pros", "cons"]).forEach((k) => { if (spec[k] && spec[k].some((l) => l && l.length)) { grid.append(h("span", { text: k === "pros" ? "Pros" : "Cons" })); spec.options.forEach((_, j) => grid.append(h("span", { text: (spec[k][j] || []).join(", ") }))); } });
  return h("div", {}, grid, h("div", { class: "l-verdictline" }, h("strong", { text: "Verdict: " }), spec.verdict || `${spec.options[spec.winner]} has the highest weighted score.`));
}

const timelineBlock = (spec) => h("ol", { class: "l-timeline" }, spec.events.map((e) => h("li", {}, h("time", { text: String(e.date) }), h("i"),
  h("span", {}, h("strong", { text: String(e.title) }), " ", e.source_ids && e.source_ids.length ? h("span", { class: "cite", text: e.source_ids.map((x) => `[${x}]`).join(" ") }) : null, e.text ? h("br") : null, e.text || ""))));
const statsBlock = (spec) => h("div", { class: "l-stats" }, spec.tiles.map((t) => h("div", {}, h("div", { class: "k", text: String(t.label) }),
  h("div", { class: "v" }, String(t.value), t.unit ? h("small", { text: " " + t.unit }) : null),
  t.delta ? h("div", { class: "d", style: { color: t.delta_good === false ? "var(--danger)" : "var(--ok)" }, text: `${/^-|−/.test(String(t.delta)) ? "▼" : "▲"} ${t.delta}${t.delta_good === false ? " (worse)" : ""}` }) : null)));
const CALLOUT = { info: ["info", "info"], warning: ["warn", "warn"], danger: ["danger", "danger"], tip: ["sparkle", "ok"] };
const calloutBlock = (spec) => { const [ic, v] = CALLOUT[spec.level] || CALLOUT.info; return h("div", { class: "l-callouts" }, h("div", { class: `l-callout v-${v}`, role: "note" }, icon(ic, 16), h("span", {}, spec.title ? h("strong", { text: spec.title + " " }) : null, spec.text))); };
function quoteBlock(spec) {
  const q = h("blockquote", { class: "l-quote" });
  const hl = spec.highlight;
  if (Array.isArray(hl) && hl.length === 2 && hl[1] > hl[0]) q.append("“" + spec.text.slice(0, hl[0]), h("mark", { text: spec.text.slice(hl[0], hl[1]) }), spec.text.slice(hl[1]) + "”");
  else q.append("“" + spec.text + "”");
  q.append(h("footer", {}, "verbatim", spec.source_id ? h("span", { class: "mono", style: { color: "var(--color-accent-text)" }, text: `[${spec.source_id}]` }) : null));
  return q;
}
const stepsBlock = (spec) => h("ol", { class: "l-steps" }, spec.steps.map((st, i) => h("li", {}, h("span", { class: `no ${st.done ? "done" : ""}`, text: st.done ? "✓" : String(i + 1) }),
  h("span", {}, h("strong", { text: String(st.title) }), st.done ? " — done" : "", st.text ? h("div", { style: { color: "var(--color-text-2)" }, text: st.text }) : null,
    st.command ? h("span", { class: "cmd" }, st.command, h("button", { "aria-label": "Copy command", onclick: () => copyText(st.command, "Command copied") }, icon("copy", 13))) : null))));
function codeBlock(spec) {
  const code = h("code", { class: spec.language ? "language-" + spec.language : "", text: spec.code });
  ensureHljs().then((hl) => { if (hl && spec.language && hl.getLanguage(spec.language)) hl.highlightElement(code); }).catch(() => {});
  const marks = new Set(spec.highlight_lines || []);
  if (marks.size) { code.textContent = ""; spec.code.split("\n").forEach((l, i) => code.append(h("span", { style: { display: "block", background: marks.has(i + 1) ? "var(--color-hl)" : "none" }, text: l || " " }))); }
  return h("div", { class: "l-code" }, h("div", { class: "l-codehead" }, h("span", { text: [spec.language, spec.filename].filter(Boolean).join(" · ") || "code" }),
    h("button", { onclick: () => copyText(spec.code, "Code copied") }, icon("copy", 12), "copy")), h("pre", {}, code));
}
const diffBlock = (spec) => h("div", {}, spec.files.map((f) => h("div", { class: "l-diff", style: { marginBottom: "6px" } }, h("div", { class: "h", text: f.path }),
  f.hunks.flatMap((hk) => (hk.lines || []).map((l) => h("div", { class: l.t || "ctx", text: ({ add: "+ ", del: "− " }[l.t] || "  ") + l.text }))))));

/** A tiny LaTeX subset (^ _ \frac \sqrt greek, \cdot, \times ...) as styled text; the source line stays visible. */
function latexToNodes(src) {
  const G = { alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", theta: "θ", lambda: "λ", mu: "μ", pi: "π", sigma: "σ", phi: "φ", omega: "ω", Delta: "Δ", Sigma: "Σ", Omega: "Ω",
    cdot: "·", times: "×", pm: "±", leq: "≤", geq: "≥", neq: "≠", approx: "≈", infty: "∞", sum: "∑", int: "∫", partial: "∂", nabla: "∇", rightarrow: "→", to: "→" };
  const out = h("span", {});
  let i = 0;
  const group = () => { if (src[i] === "{") { let d = 1, j = i + 1; while (j < src.length && d) { if (src[j] === "{") d++; else if (src[j] === "}") d--; j++; } const g = src.slice(i + 1, j - 1); i = j; return g; } return src[i++] || ""; };
  while (i < src.length) {
    const c = src[i];
    if (c === "^" || c === "_") { i++; out.append(h(c === "^" ? "sup" : "sub", { style: { fontSize: ".6em" }, text: group() })); }
    else if (c === "\\") {
      const m = /^[A-Za-z]+/.exec(src.slice(i + 1)); const name = m ? m[0] : ""; i += 1 + name.length;
      if (name === "frac") { const a = group(), b = group(); out.append(`(${a})/(${b})`); }
      else if (name === "sqrt") out.append("√(" + group() + ")");
      else out.append(G[name] || name);
    } else if (c === "{" || c === "}") i++;
    else { out.append(c); i++; }
  }
  return out;
}
const mathBlock = (spec) => h("div", {}, h("div", { class: "l-math", role: "img", "aria-label": spec.latex }, latexToNodes(spec.latex)), h("div", { class: "mono", style: { fontSize: "11px", color: "var(--color-text-3)" }, text: "LaTeX: " + spec.latex + " · rendered offline, no CDN" }));
const mapBlock = (spec) => h("div", { style: { display: "flex", flexDirection: "column", gap: "8px" } },
  h("div", { class: "l-note v-info", style: { padding: "4px 8px", borderRadius: "var(--radius-md)", alignItems: "center" } }, icon("wifioff", 14), "Map tiles need the internet. Showing coordinates instead."),
  spec.points.map((p) => h("div", { style: { display: "flex", justifyContent: "space-between", fontSize: "14px", padding: "6px 0" } }, h("span", {}, icon("pin", 14), " " + (p.label || "point")),
    h("span", { class: "mono", text: `${Math.abs(p.lat).toFixed(3)}° ${p.lat >= 0 ? "N" : "S"}, ${Math.abs(p.lon).toFixed(3)}° ${p.lon >= 0 ? "E" : "W"}` }))));
const galleryBlock = (spec) => h("div", { class: "l-gallery" }, spec.images.map((i) => h("figure", {}, h("img", { src: i.url, alt: i.caption || "", loading: "lazy" }),
  i.caption || i.attribution ? h("figcaption", {}, i.caption || "", i.attribution ? h("span", { style: { color: "var(--color-text-3)" }, text: ` ${i.attribution}${i.source_id ? " · [" + i.source_id + "]" : ""}` }) : null) : null)));
const svgBlock = (spec) => h("img", { class: "l-svgblock", alt: spec.alt || "illustration", src: "data:image/svg+xml;utf8," + encodeURIComponent(spec.svg) });
const htmlBlock = (spec) => h("div", { class: "l-htmlbox" }, h("div", { class: "cap" }, icon("lock", 14), h("strong", { text: "Interactive content" }), " — runs in a sandbox with no network."),
  h("iframe", { sandbox: "allow-scripts", title: "Interactive content", style: { height: Math.min(Math.max(parseInt(spec.height) || 360, 120), 900) + "px" },
    srcdoc: `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">` + spec.html }));

const RENDER = { chart: chartBlock, table: tableBlock, diagram: diagramBlock, compare: compareBlock, timeline: timelineBlock, stats: statsBlock, callout: calloutBlock, quote: quoteBlock,
  steps: stepsBlock, code: codeBlock, diff: diffBlock, math: mathBlock, map: mapBlock, gallery: galleryBlock, svg: svgBlock, html: htmlBlock };

// ------------------------------------------------------------------------------------------------ frame, tools, states

const PROV = { sources: ["link", (b) => (b.source_ids && b.source_ids.length ? `data from ${b.source_ids.map((n) => `[${n}]`).join(", ")}` : "data from the sources")],
  computed: ["cpu", () => "computed with the calculator"], model: ["warn", () => "model knowledge — not checked against a source"] };

function tools(block, body, { narrow, onPin, boardId }) {
  const actions = [
    { i: "copy", l: "Copy", f: () => copyText(body.csv ? body.csv() : JSON.stringify(block.spec, null, 2), "Copied") },
    { i: "download", l: "Export", f: () => {
      const name = (block.title || block.kind).replace(/[^\w-]+/g, "_");
      if (body.csv) download(name + ".csv", body.csv(), "text/csv");
      else if (body.svg && body.svg()) download(name + ".svg", body.svg(), "image/svg+xml");
      else if (block.kind === "svg") download(name + ".svg", block.spec.svg, "image/svg+xml");
      else download(name + ".json", JSON.stringify(block.spec, null, 2), "application/json");
    } },
    { i: "expand", l: "Expand", f: () => dialog({ title: block.title || block.kind, body: h("div", { style: { maxHeight: "70vh", overflow: "auto" } }, RENDER[block.kind](block.spec)), actions: [{ id: "x", label: "Close", primary: true }] }) },
    { i: "pin", l: boardId ? "Remove from Board" : "Pin to Board", f: onPin },
    { i: "code", l: "Show spec", f: () => { const pre = body.parentElement.querySelector(".spec"); if (pre) pre.classList.toggle("hidden"); } },
  ];
  const btn = (a) => h("button", { class: "tb", "aria-label": a.l, title: a.l, onclick: a.f }, icon(a.i, 16));
  if (!narrow) return actions.map(btn);
  const more = h("div", { class: "hidden", style: { display: "flex", gap: "4px", padding: "0 var(--space-3)" } }, actions.filter((a) => a.i !== "pin").map(btn));
  return [btn(actions[3]), h("button", { class: "tb", "aria-label": "More: copy, export, expand, spec", onclick: () => more.classList.toggle("hidden") }, icon("dots", 16)), more];
}

export function renderBlock(block, { narrow = matchMedia("(max-width: 820px)").matches, board = false, boardId = null, updated = "", onPin = null, onRefresh = null } = {}) {
  const [ic] = KIND[block.kind] || ["info", block.kind];
  const fig = h("figure", { class: "l-block", dataset: { block: block.id || "", kind: block.kind } });
  if (block.kind === "error") return blockError(block.title, block.spec && block.spec.reason, block.spec && block.spec.raw);
  let body;
  try { body = RENDER[block.kind](block.spec); } catch (e) { return blockError(block.title, `The ${block.kind} data is not valid (${e.message}).`, JSON.stringify(block.spec, null, 2)); }
  const pin = onPin || (async () => {
    try { await api.post("/api/board/items", { block }); toast("Pinned to the Board"); }
    catch (e) { toast(e.status === 404 ? "The Board isn’t available yet" : e.detail || "Could not pin"); }
  });
  const wrap = h("div", { class: "bd" }, body, h("pre", { class: "raw spec hidden", text: JSON.stringify(block.spec, null, 2) }));
  const prov = PROV[block.provenance] || PROV.sources;
  append(fig, [
    h("div", { class: "hd" }, icon(ic, 16), h("strong", { text: block.title || (KIND[block.kind] || [])[1] || block.kind }), ...tools(block, body, { narrow, onPin: pin, boardId })),
    block.provenance === "model" ? h("div", { class: "l-unverified", role: "note" }, icon("warn", 14), h("strong", { text: "Unverified" }), " — from the model’s own knowledge, not from a source.") : null,
    wrap,
    h("figcaption", { class: `ft ${block.provenance === "model" ? "model" : ""}` }, icon(prov[0], 13), h("span", { text: prov[1](block) }),
      board ? h("span", { class: "r" }, h("span", { class: "mono", text: updated ? "updated " + updated : "" }), onRefresh ? h("button", { class: "tb", "aria-label": "Refresh", title: "Refresh", style: { width: "28px", height: "28px" }, onclick: onRefresh }, icon("refresh", 14)) : null) : null)]);
  return fig;
}

/** Placeholder while the model is still writing the tool call (space is reserved so nothing jumps). */
export function blockSkeleton(kind) {
  const [, word] = KIND[kind] || ["", "block"];
  const t0 = Date.now(), clock = h("span", { class: "mono", text: "0 s" });
  const fig = h("figure", { class: "l-block", role: "status", style: { minHeight: "150px" } },
    h("div", { class: "bd", style: { display: "flex", flexDirection: "column", gap: "10px", justifyContent: "center" } },
      h("div", { class: "l-skeleton" }, h("i"), h("i"), h("i")),
      h("div", { style: { fontSize: "13px", color: "var(--color-text-2)", display: "flex", gap: "8px", alignItems: "center" } }, icon("clock", 14), `preparing ${word}… `, clock,
        h("span", { style: { color: "var(--color-text-3)" }, text: "· space is reserved, nothing will jump" }))));
  const timer = setInterval(() => { if (!fig.isConnected) return clearInterval(timer); clock.textContent = Math.round((Date.now() - t0) / 1000) + " s"; }, 1000);
  return fig;
}

export function blockError(title, reason, raw) {
  return h("figure", { class: "l-block err" }, h("div", { class: "hd" }, icon("danger", 16), h("strong", { text: title || "Block could not be drawn" })),
    h("div", { class: "bd", style: { display: "flex", flexDirection: "column", gap: "8px" } },
      h("div", { class: "why" }, icon("danger", 16), h("span", {}, h("strong", { text: "Couldn’t draw this. " }), reason || "The data is invalid.")),
      raw ? h("details", {}, h("summary", { class: "l-link", text: "show raw spec" }), h("pre", { class: "raw", text: typeof raw === "string" ? raw : JSON.stringify(raw, null, 2) })) : null));
}
