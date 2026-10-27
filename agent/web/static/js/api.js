// REST + SSE client. Adds the access token, turns HTTP failures into ApiError, and tells the app when the token is
// missing/rejected or the server cannot be reached (the Connect screen listens).

import { set } from "./store.js";

const TOKEN_KEY = "agent_token";
export const getToken = () => { try { return localStorage.getItem(TOKEN_KEY) || ""; } catch (_) { return ""; } };
export const setToken = (t) => { try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch (_) { /* private mode */ } };

export class ApiError extends Error {
  constructor(status, detail) { super(detail || `HTTP ${status}`); this.status = status; this.detail = detail || ""; }
}

function headers(extra = {}) {
  const t = getToken();
  return { ...(t ? { Authorization: "Bearer " + t } : {}), ...extra };
}

async function failure(res) {
  let detail = "";
  try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || j); } catch (_) { /* no body */ }
  if (res.status === 401) set({ connection: getToken() ? "expired" : "unauthorized" });
  return new ApiError(res.status, detail);
}

async function request(method, path, body) {
  let res;
  try {
    res = await fetch(path, { method, headers: headers(body === undefined ? {} : { "Content-Type": "application/json" }), body: body === undefined ? undefined : JSON.stringify(body) });
  } catch (e) {
    set({ connection: "offline" });
    throw new ApiError(0, "Cannot reach the server.");
  }
  if (!res.ok) throw await failure(res);
  set({ connection: "ok" });
  return res.status === 204 ? null : res.json();
}

export const api = {
  get: (p) => request("GET", p),
  post: (p, b) => request("POST", p, b === undefined ? {} : b),
  put: (p, b) => request("PUT", p, b),
  patch: (p, b) => request("PATCH", p, b),
  del: (p) => request("DELETE", p),
  /** A /api/files/<token> URL (capability URL: no header needed, so <img>/<audio> can use it). */
  file: (url) => url,
};

/** PUT a file as the raw request body (attachments). Resolves with the server's JSON ({path, name, size}). */
export async function upload(path, file) {
  let res;
  try { res = await fetch(path, { method: "PUT", headers: headers({ "Content-Type": file.type || "application/octet-stream" }), body: file }); }
  catch (e) { set({ connection: "offline" }); throw new ApiError(0, "Cannot reach the server."); }
  if (!res.ok) throw await failure(res);
  return res.json();
}

/**
 * POST (or GET) a Server-Sent-Events endpoint and call handlers[type](event) for every event, in order.
 * Returns a promise that settles when the stream ends; `signal` (AbortController) cancels it.
 */
export async function stream(path, body, handlers = {}, signal) {
  let res;
  try {
    res = await fetch(path, { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify(body ?? {}), signal });
  } catch (e) {
    if (e.name === "AbortError") return;
    set({ connection: "offline" });
    throw new ApiError(0, "Connection lost.");
  }
  if (!res.ok) throw await failure(res);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop();
      for (const chunk of chunks) {
        let data = null;
        for (const line of chunk.split("\n")) if (line.startsWith("data: ")) { try { data = JSON.parse(line.slice(6)); } catch (_) { data = null; } }
        if (!data || !data.type) continue;
        (handlers[data.type] || handlers.default || (() => {}))(data);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") throw e;
  }
}

export async function checkAuth() {
  try { await api.get("/api/auth/check"); return true; } catch (e) { return false; }
}
