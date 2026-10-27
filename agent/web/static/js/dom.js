// Tiny DOM helpers. Everything that comes from the server goes in through textContent or attributes, never innerHTML.

// Safety net: the DOM turns `append(null)` into the text "null". Conditional children (`cond ? el : null`) are everywhere, so
// the node-adding methods skip null / undefined / false for the whole app.
for (const name of ["append", "prepend", "replaceChildren"]) {
  const original = Element.prototype[name];
  Element.prototype[name] = function (...nodes) { return original.apply(this, nodes.filter((n) => n !== null && n !== undefined && n !== false)); };
}

export function h(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) node.setAttribute(k, "");
    else node.setAttribute(k, v);
  }
  append(node, kids);
  return node;
}

export function append(node, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

export const clear = (node) => { node.replaceChildren(); return node; };

export function svg(tag, attrs = {}, ...kids) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const kid of kids.flat()) node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return node;
}

export const mono = (text, cls = "") => h("span", { class: ("mono " + cls).trim(), text });

/** A date/time like "18 Oct 2026 14:03" in the user's locale, or "" for a missing value. */
export function fmtDateTime(value) {
  const d = value instanceof Date ? value : new Date(typeof value === "number" && value < 1e12 ? value * 1000 : value);
  if (isNaN(d)) return "";
  return d.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function relTime(value) {
  const t = (value instanceof Date ? value : new Date(typeof value === "number" && value < 1e12 ? value * 1000 : value)).getTime();
  if (isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "now";
  if (s < 3600) return Math.round(s / 60) + " min";
  if (s < 86400) return Math.round(s / 3600) + " h";
  if (s < 172800) return "yesterday";
  return new Date(t).toLocaleDateString(undefined, { weekday: "short" });
}

export const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
