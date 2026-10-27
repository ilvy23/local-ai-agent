// A very small observable store for what several screens share: theme, density, connection, model, host, context.

const KEY = "lantern.prefs";
const defaults = { theme: "dark", density: "comfortable" };

function load() {
  try { return { ...defaults, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch (_) { return { ...defaults }; }
}

const listeners = new Set();
export const state = {
  ...load(),
  connection: "ok",            // ok | offline | unauthorized | asleep | ollama | restarting | nomodel | expired
  model: { name: "", state: "unknown", detail: "" },      // state: loaded | loading | unloaded | error | unknown
  host: { state: "unknown", text: "" },                    // state: ok | busy | paused | asleep | unknown
  ctx: null,                   // { used, limit, percent, tokens_per_s } of the open chat
  sessionId: null,
};

export function set(patch) {
  Object.assign(state, patch);
  if ("theme" in patch || "density" in patch) {
    try { localStorage.setItem(KEY, JSON.stringify({ theme: state.theme, density: state.density })); } catch (_) { /* private mode */ }
  }
  for (const fn of listeners) fn(state, patch);
}

export function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }

/** "light" | "dark": the explicit choice, else the system preference. */
export const effectiveTheme = () =>
  state.theme === "auto" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : state.theme;

export function applyTheme() {
  const root = document.documentElement;
  root.setAttribute("data-theme", effectiveTheme());
  root.setAttribute("data-density", state.density);
  const dark = effectiveTheme() === "dark";
  const link = document.getElementById("hljs-theme");
  if (link) link.setAttribute("href", dark ? "/vendor/hljs-dark.css" : "/vendor/hljs-light.css");
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", dark ? "#0a0a0d" : "#f6f5fa");
}
