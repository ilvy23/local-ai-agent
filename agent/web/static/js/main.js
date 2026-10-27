// Boot: theme, auth check, shell, and a hash router that loads one screen module at a time (js/screens/<name>.js).
import { applyTheme, state, subscribe } from "./store.js";
import { createShell, TITLES, addNavFeatures } from "./shell.js";
import { loadFeatures } from "./features.js";
import { checkAuth } from "./api.js";
import { h } from "./dom.js";

applyTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);

const shell = createShell(document.getElementById("app"));
const SCREENS = new Set(["chat", "board", "memory", "skills", "projects", "code", "models", "status", "audit", "whatsapp", "prompt", "settings", "system", "more", "sessions", "knowledge", "apps", "research", "tools", "ops", "patterns"]);
let cleanups = [], token = 0, current = "", featuresDone = false;

const loadedCss = new Set();
function css(name) {
  if (loadedCss.has(name)) return;
  loadedCss.add(name);
  document.head.append(h("link", { rel: "stylesheet", href: `/css/${name}.css` }));
}

function parse() {
  const [name = "chat", id = ""] = (location.hash.replace(/^#\/?/, "") || "chat").split("/");
  return { name: SCREENS.has(name) ? name : "chat", params: id ? { id } : {} };
}

async function route() {
  const mine = ++token;
  for (const fn of cleanups.splice(0)) { try { fn(); } catch (_) { /* a screen's cleanup must not block navigation */ } }
  const locked = ["unauthorized", "expired"].includes(state.connection);
  if (!locked && !featuresDone) {                              // optional features (screens that exist only in some builds) join the navigation first
    const list = await loadFeatures();
    if (list) { addNavFeatures(list).forEach((k) => SCREENS.add(k)); featuresDone = true; }
    if (mine !== token) return;
  }
  const r = parse();
  const name = locked ? "system" : r.name;
  shell.outlet.replaceChildren();
  shell.setRoute({ name, params: r.params, title: TITLES[name] });
  current = name;
  let mod;
  try { mod = await import(`./screens/${name}.js`); }
  catch (e) {
    console.error("screen failed to load", name, e);
    shell.outlet.replaceChildren(h("div", { class: "l-page" }, h("h2", { text: TITLES[name] || name }), h("div", { class: "l-state" }, "This screen could not be loaded.", h("span", { class: "mono", text: String(e && e.message || e) }))));
    return;
  }
  if (mine !== token) return;                                   // another navigation started while the module loaded
  const root = h("div", { class: "l-screen" });
  shell.outlet.replaceChildren(root);
  const ctx = { name, params: r.params, shell, css, onCleanup: (fn) => cleanups.push(fn), navigate: (hash) => { location.hash = hash; }, route };
  try { await mod.mount(root, ctx); }
  catch (e) {
    console.error("screen crashed", name, e);
    root.replaceChildren(h("div", { class: "l-page" }, h("div", { class: "l-state" }, "Something went wrong in this screen.", h("span", { class: "mono", text: String(e && e.message || e) }))));
  }
}

window.addEventListener("hashchange", route);
subscribe((s, patch) => { if ("connection" in patch && (["unauthorized", "expired"].includes(patch.connection) || (current === "system" && patch.connection === "ok"))) route(); });
checkAuth().finally(route);
