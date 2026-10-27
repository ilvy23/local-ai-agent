// Optional features the server reports at /api/features (screens and tabs that exist only when their backend module is
// installed). Loaded once; a failed request (not signed in yet, server down) gives null and is retried on the next call.
import { api } from "./api.js";

let list = null;

export async function loadFeatures() {
  if (list) return list;
  try { const r = await api.get("/api/features"); list = Array.isArray(r) ? r : []; } catch (_) { return null; }
  return list;
}

/** Tabs that features add to one screen, as modules (each exports `tab(ctx, getView)` -> { k, l, load, poll? }). */
export async function featureTabs(screen, ctx, getView) {
  const out = [];
  for (const f of (await loadFeatures()) || []) {
    if (f.tab_of !== screen || !f.module) continue;
    try { out.push((await import(`./${f.module}.js`)).tab(ctx, getView)); } catch (e) { console.error("feature tab failed to load", f.key, e); }
  }
  return out;
}
