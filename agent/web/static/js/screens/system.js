// Connection screen (design S-System): connect with the access token, bad token, PC asleep (wake gate), model server
// unreachable, model not installed, server restarting, session expired, offline. Each: a calm explanation, one action.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { mark } from "../mark.js";
import { api, setToken } from "../api.js";
import { state } from "../store.js";

// state -> [icon, variant, tag, title, text, button, button icon, foot]
const STATES = {
  connect: ["lock", "info", "First run", "Connect to your PC", "Enter the access token shown on the PC. It stays on this device.", "Connect", "arrow", "Reached over your private Tailscale network. No internet needed."],
  badtoken: ["danger", "danger", "Token rejected", "That token didn’t work", "It may be mistyped or expired. Create a new one on the PC and enter it again.", "Try again", "refresh", "Nothing was sent anywhere except your PC."],
  asleep: ["power", "warn", "PC asleep", "Your PC is asleep", "Waking it takes about 20 seconds; chats are safe.", "Wake up PC", "power", "Sends a wake signal over your tailnet."],
  ollama: ["danger", "danger", "Model server unreachable", "The model server isn’t answering", "Ollama didn’t respond on the PC. The app is fine; sending is paused.", "Try again", "refresh", ""],
  nomodel: ["info", "warn", "Model not installed", "The chat model isn’t installed", "Pick an installed model to keep chatting, or install this one on the PC.", "Choose a model", "cpu", ""],
  restarting: ["refresh", "info", "Server restarting", "The server is restarting", "This usually takes under a minute. We’ll reconnect by ourselves.", "Reconnecting…", "clock", "Your draft is kept."],
  expired: ["clock", "warn", "Session expired", "Your session ended", "For safety the connection times out. Enter the token again to continue — your chats are still here.", "Reconnect", "arrow", "Nothing was lost."],
  offline: ["wifioff", "neutral", "Offline", "This phone left the tailnet", "Open Tailscale and make sure it’s connected. Your drafts are saved on this device.", "Retry", "refresh", "Retrying every 10 seconds."],
  ok: ["check", "ok", "Connected", "Everything is reachable", "The server, the model server and the chat model all answer.", "Back to chat", "arrow", ""],
};

// Survives the re-mount that api.js triggers (connection -> unauthorized/expired) when a token is rejected.
let rejected = "";

/** true = awake, false = the wake host says asleep, null = no wake service here (404, network error, bad body). */
async function wakeStatus() {
  try {
    const res = await fetch("/_wake_status", { cache: "no-store" });
    if (!res.ok) return null;
    const j = await res.json();
    return typeof j.awake === "boolean" ? j.awake : null;
  } catch (_) { return null; }
}

/** Work out which state describes reality, from real signals. Returns { state, installed? }. */
async function diagnose() {
  if ((await wakeStatus()) === false) return { state: "asleep" };
  let st, models;
  try { st = await api.get("/api/status"); }
  catch (e) {
    if (e.status === 0) return { state: "offline" };
    if ([502, 503, 504].includes(e.status)) return { state: "restarting" };
    if (e.status === 401) return { state: state.connection === "expired" ? "expired" : "connect" };
    return { state: "ollama" };
  }
  try { models = await api.get("/api/models"); } catch (_) { return { state: "ollama" }; }
  const installed = models.available || [];
  if (!installed.length && !(st.loaded_models || []).length) return { state: "ollama" };
  const want = String(models.current || st.chat_model || "");
  if (want && !installed.includes(want)) return { state: "nomodel", want, installed };
  return { state: "ok" };
}

export async function mount(root, ctx) {
  ctx.css("s-system");
  ctx.shell.setTitle("Connection");
  let timer = null, alive = true;
  const stopTimer = () => { clearInterval(timer); timer = null; };
  ctx.onCleanup(() => { alive = false; stopTimer(); });

  const page = h("div", { class: "s-system" });
  root.replaceChildren(page);

  function view(name, { error = "", foot = "", alt = null, busy = false, onPrimary, field = false, title = "" } = {}) {
    const [ic, v, tag, ttl, text, btn, btnIcon, defFoot] = STATES[name];
    const token = h("input", { id: "s-system-tok", class: "input s-system-in", type: "password", autocomplete: "off", autocapitalize: "off", spellcheck: "false",
      "aria-describedby": error ? "s-system-err" : null, "aria-invalid": error ? "true" : null });
    const go = (e) => { e && e.preventDefault(); onPrimary && onPrimary(token.value.trim()); };
    const primary = h("button", { class: "btn btn-primary s-system-btn", type: "submit", disabled: busy }, icon(btnIcon, 16), btn);
    const form = h("form", { class: "s-system-form", novalidate: true, onsubmit: go },
      field ? h("div", { class: "s-system-fields" },
        h("div", { class: "field" }, h("label", { for: "s-system-addr", text: "Address" }), h("input", { id: "s-system-addr", class: "input s-system-in", readonly: true, value: location.host })),
        h("div", { class: "field" }, h("label", { for: "s-system-tok", text: "Access token" }), token,
          error ? h("div", { id: "s-system-err", class: "s-system-err", role: "alert" }, icon("danger", 14), error) : null)) : null,
      h("div", { class: "s-system-actions" }, primary, alt ? h("button", { class: "btn btn-ghost s-system-btn", type: "button", onclick: alt[1] }, alt[0]) : null));
    if (!field && error) form.prepend(h("div", { class: "s-system-err", role: "alert" }, icon("danger", 14), error));
    page.replaceChildren(h("div", { class: "s-system-card" }, mark(40, false),
      h("div", { class: `s-system-chip v-${v}${v === "neutral" ? " dashed" : ""}` }, icon(ic, 14), tag),
      h("h2", { text: title || ttl }), h("p", { class: "s-system-text", text }), form,
      (foot || defFoot) ? h("p", { class: "s-system-foot", text: foot || defFoot }) : null));
    if (field) token.focus();
    return token;
  }

  async function connect(value) {
    if (!value) return show("badtoken", { error: "Enter the access token first." });
    setToken(value);
    rejected = "Invalid or expired token (401).";  // set first: api.js re-mounts this screen on a 401, before the catch below runs
    show(state.connection === "expired" ? "expired" : "connect", { busy: true });
    try {
      await api.get("/api/auth/check");           // api.js sets connection "ok"; main.js then re-routes
      rejected = "";
    } catch (e) {
      setToken("");                               // a rejected token is not worth keeping
      if (e.status !== 401) rejected = "";
      if (alive && e.status !== 401) show(e.status === 0 ? "offline" : "badtoken", { error: e.status === 0 ? "" : (e.detail || `Server answered ${e.status}.`) });
    }
  }

  async function wake() {
    let ticks = 0;
    show("asleep", { busy: true, foot: "This can take up to about 30 seconds.", title: "Waking up…" });
    try { await fetch("/_wake", { method: "POST" }); } catch (_) { /* still poll: the click is what matters */ }
    stopTimer();
    timer = setInterval(async () => {
      if (!alive) return stopTimer();
      if ((await wakeStatus()) !== false || ++ticks > 60) { stopTimer(); check(); }
    }, 2000);
  }

  async function check() {
    show("ok", { busy: true, title: "Checking…" });
    const r = await diagnose();
    if (alive) show(r.state, { want: r.want, installed: r.installed });
  }

  /** Render one state with its behaviour. */
  function show(name, o = {}) {
    stopTimer();
    if (!alive) return;
    const back = () => ctx.navigate("#/chat");
    const again = { onPrimary: check };
    switch (name) {
      case "connect": case "badtoken": case "expired":
        return view(name, { field: true, onPrimary: connect, ...o });
      case "asleep": return view(name, { onPrimary: wake, ...o });
      case "ollama": return view(name, { ...again, alt: ["Open host status", () => ctx.navigate("#/status")], ...o });
      case "nomodel": return view(name, { onPrimary: () => ctx.navigate("#/models"), title: o.want ? `${o.want} isn’t installed` : "",
        foot: o.installed && o.installed.length ? "Installed: " + o.installed.join(", ") : "", ...o });
      case "restarting":
        timer = setInterval(async () => { try { await api.get("/api/status"); stopTimer(); check(); } catch (_) { /* keep waiting */ } }, 3000);
        return view(name, { busy: true, ...o });
      case "offline":
        timer = setInterval(check, 10000);
        return view(name, { ...again, ...o });
      case "ok": return view(name, { onPrimary: back, ...o });
      default: return view("connect", { field: true, onPrimary: connect });
    }
  }

  if (state.connection === "unauthorized") show(rejected ? "badtoken" : "connect", rejected ? { error: rejected } : {});
  else if (state.connection === "expired") show(rejected ? "badtoken" : "expired", rejected ? { error: rejected } : {});
  else check();
}
