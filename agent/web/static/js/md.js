// Markdown -> DOM for assistant replies. marked does not sanitise, so the parsed HTML is cleaned in an inert template
// (scripts, handlers, javascript: URLs removed) before insertion. Code blocks get a header with a copy button, tables get a scroll wrapper, and "[n]" becomes a tappable citation.
import { h } from "./dom.js";
import { icon } from "./icons.js";
import { ensureMarked, ensureHljs, ensureMermaid } from "./vendor.js";
import { copyText } from "./ui.js";
import { effectiveTheme } from "./store.js";

const CITE = /\[(\d{1,2})(?:\s*[,;]\s*\d{1,2})*\]/g;

function citations(root, onCite, unsupported) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) => (n.parentElement.closest("code, pre, a, button") ? NodeFilter.FILTER_REJECT : CITE.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT) });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const frag = document.createDocumentFragment();
    let last = 0;
    node.nodeValue.replace(CITE, (match, _n, offset) => {
      frag.append(document.createTextNode(node.nodeValue.slice(last, offset)));
      for (const num of match.match(/\d+/g)) frag.append(h("button", { class: "l-cite", "aria-label": `Source ${num}`, onclick: () => onCite && onCite(Number(num)), text: `[${num}]` }));
      last = offset + match.length;
      return match;
    });
    frag.append(document.createTextNode(node.nodeValue.slice(last)));
    node.replaceWith(frag);
  }
}

/** Mark sentences the answer checker could not ground in a source (verdict.claims with status "unsupported"). */
function markUnsupported(root, claims) {
  for (const c of claims || []) {
    if (c.status !== "unsupported") continue;
    const needle = (c.text || "").replace(/\s*\[\d+\]\s*/g, " ").trim().slice(0, 60);
    if (!needle) continue;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const n = walker.currentNode, i = n.nodeValue.indexOf(needle);
      if (i < 0 || n.parentElement.closest("button, code, pre")) continue;
      const mid = n.splitText(i); mid.splitText(needle.length);
      mid.replaceWith(h("span", { class: "l-claim", title: "No source found for this statement" }, mid.nodeValue, icon("warn", 14), h("b", { text: "no source" })));
      break;
    }
  }
}

/** Mark figures that appear in no source ("43 % faster" + a warning icon), as in the design. The number, its unit and the next word. */
function markStray(root, figures) {
  for (const v of figures || []) {
    const n = String(v).replace(/\.0+$/, "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(?<![\\d.,])${n}(?![\\d])(\\s*(?:%|€|\\$|£|[A-Za-z]+))?(\\s+[A-Za-zÄÖÜäöüß]+)?`);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode, m = re.exec(node.nodeValue);
      if (!m || node.parentElement.closest("button, code, pre, .l-claim")) continue;
      const mid = node.splitText(m.index); mid.splitText(m[0].length);
      mid.replaceWith(h("span", { class: "l-claim", title: "This figure appears in none of the sources read" }, mid.nodeValue, icon("warn", 14), h("b", { text: "no source" })));
      break;
    }
  }
}

export async function renderMarkdown(text, { onCite = null, claims = null, stray = null } = {}) {
  const box = h("div", { class: "l-prose" });
  let marked;
  try { marked = await ensureMarked(); } catch (_) { box.textContent = text; return box; }
  const html = marked.parse(text || "", { gfm: true, breaks: false, async: false });
  const tpl = document.createElement("template");
  tpl.innerHTML = html;                                    // a <template> is inert: nothing runs until it is inserted
  const frag = tpl.content;                                // ...so it is cleaned BEFORE it goes into the page
  frag.querySelectorAll("script, iframe, object, embed, style, link, meta, form, input, button, textarea, base").forEach((n) => n.remove());
  frag.querySelectorAll("*").forEach((n) => {              // no event handlers, no javascript: / data: URLs on links
    for (const a of [...n.attributes]) {
      if (a.name.startsWith("on") || (["href", "src", "xlink:href", "srcset", "action", "formaction"].includes(a.name) && /^\s*(javascript|vbscript):/i.test(a.value))) n.removeAttribute(a.name);
    }
    if (n.tagName === "A" && /^\s*data:/i.test(n.getAttribute("href") || "")) n.removeAttribute("href");
  });
  box.append(frag);
  box.querySelectorAll("a[href]").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
  box.querySelectorAll("table").forEach((t) => { const w = h("div", { class: "l-tablewrap" }); t.replaceWith(w); w.append(t); });
  const hljs = await ensureHljs().catch(() => null);
  for (const code of box.querySelectorAll("pre > code")) {
    const pre = code.parentElement, lang = (code.className.match(/language-([\w+-]+)/) || [])[1] || "";
    if (lang === "mermaid") {
      try {
        const m = await ensureMermaid(effectiveTheme() === "dark");
        const { svg } = await m.render("mmd" + Math.random().toString(36).slice(2), code.textContent);
        const holder = h("div", { class: "l-code", style: { padding: "var(--space-2)" } });
        holder.innerHTML = svg;                           // mermaid runs in securityLevel "strict": its SVG is sanitised
        pre.replaceWith(holder);
        continue;
      } catch (_) { /* fall through: show the source */ }
    }
    if (hljs && lang && hljs.getLanguage(lang)) hljs.highlightElement(code);
    const raw = code.textContent;
    pre.prepend(h("div", { class: "l-codehead" }, h("span", { text: lang || "code" }), h("button", { onclick: () => copyText(raw, "Code copied"), "aria-label": "Copy code" }, icon("copy", 12), "copy")));
  }
  citations(box, onCite);
  markUnsupported(box, claims);
  markStray(box, stray);
  return box;
}
