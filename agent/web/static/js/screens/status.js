// Status (design S-Status): host stats, the background worker and the model servers, read live. The only action is
// suspending the PC; worker and server control stays in the terminal.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, confirmDialog } from "../ui.js";
import { pill, loadingBox, errorBox, meter, commandBox, sectionHead, gb, dash } from "../components/ops-ui.js";

const POLL_MS = 5000;
const WORKER = { running: ["ok", "check", "running"], paused: ["warn", "clock", "paused"], stopped: ["neutral", "stop", "stopped"], unknown: ["neutral", "info", "unknown"] };
const line = (label, value) => h("div", { class: "s-status-line" }, h("span", { text: label }), h("span", { text: value }));

export async function mount(root, ctx) {
  ctx.css("ops-ui"); ctx.css("s-status");
  ctx.shell.setTitle("Status");
  const page = h("div", { class: "l-page s-status" });
  root.replaceChildren(page);
  let timer = null, alive = true, suspending = false, first = true;
  ctx.onCleanup(() => { alive = false; clearInterval(timer); });

  async function suspend() {
    if (!(await confirmDialog("Suspend this PC?", "The PC goes to sleep now. Chat and background work stop, and this app can’t reach it until you wake it from the phone’s wake screen.", { confirm: "Suspend PC", danger: true }))) return;
    suspending = true; render();
    try { await api.post("/api/host/suspend"); toast("Suspending the PC…"); }
    catch (e) { suspending = false; toast(e.detail || "Could not suspend the PC", 4200); render(); }
  }

  let snap = null;
  function render() {
    const { status, detail, host, at } = snap;
    const game = (detail && detail.game) || (status && status.game_detected);
    const paused = (detail && detail.paused_reason) || (status && status.paused_reason);
    const gpu = host && host.gpu, ram = host && host.ram;
    const ban = game
      ? h("div", { class: "s-status-ban v-warn", role: "status" }, icon("warn", 18), h("span", {}, h("strong", { text: "Game detected." }), ` Background workers are paused so the game keeps the GPU.${typeof game === "string" ? ` (${game})` : ""}`))
      : paused ? h("div", { class: "s-status-ban v-warn", role: "status" }, icon("clock", 18), h("span", {}, h("strong", { text: "Background work is paused." }), ` ${paused}`)) : null;

    const vram = gpu ? gpu.vram_used_mb / (gpu.vram_total_mb || 1) : 0;
    const gpuCard = h("div", { class: "l-card s-status-card" }, h("div", { class: "s-status-cap", text: "GPU" }),
      gpu ? [h("strong", { class: "s-status-big", text: `${gpu.name} · ${gb(gpu.vram_total_mb)} GB` }),
        line("VRAM", `${gb(gpu.vram_used_mb)} / ${gb(gpu.vram_total_mb)} GB`), meter(vram, vram > 0.85 ? "warn" : "accent"),
        h("div", { class: "s-status-gpu" }, [["Temp", dash(gpu.temp_c, " °C")], ["Fan", dash(gpu.fan_percent, " %")], ["Load", dash(gpu.utilization_percent, " %")],
          ["Power", gpu.power_watts == null ? "—" : `${gpu.power_watts}${gpu.power_limit_watts ? `/${gpu.power_limit_watts}` : ""} W`]]
          .map(([l, v]) => h("div", {}, h("div", { text: l }), h("div", { text: v }))))]
        : h("div", { class: "s-status-nogpu" }, icon("shield-q", 18), host ? "No GPU data. The driver didn’t answer." : "No GPU data. The host stats didn’t load."));

    const cpuCard = h("div", { class: "l-card s-status-card" }, h("div", { class: "s-status-cap", text: "CPU · RAM" }),
      line("CPU load", `${dash(host && host.cpu_load_percent, " %")} · ${dash(host && host.cpu_temp_c, " °C")}`),
      line("RAM", ram ? `${ram.used_gb} / ${ram.total_gb} GB` : "—"), ram ? meter(ram.used_gb / (ram.total_gb || 1)) : null);

    const loaded = (status && status.loaded_models) || [];
    const hostCard = h("div", { class: "l-card s-status-card" }, h("div", { class: "s-status-cap", text: "Host" }),
      line("Local IP", dash(host && host.local_ip)), line("Uptime", dash(host && host.uptime)),
      status ? line("GPU mode", dash(status.gpu_mode)) : null, status ? line("Chat model", dash(status.chat_model)) : null,
      h("button", { class: "btn btn-secondary", style: { alignSelf: "flex-start", minHeight: "44px" }, disabled: suspending, onclick: suspend },
        icon("power", 15), suspending ? "Suspending…" : "Suspend PC…"));

    const w = detail && detail.worker;
    let workerRow;
    if (w) {
      const [tone, ic, word] = WORKER[w.state] || WORKER.unknown;
      const why = w.state === "paused" ? w.reason : w.activity ? `${w.activity}${w.activity_ago_s != null ? ` · ${w.activity_ago_s} s ago` : ""}` : w.state === "unknown" ? "systemd did not answer" : "";
      workerRow = h("div", { class: "ops-row s-status-row" }, pill(tone, ic, word), h("strong", { class: "s-status-name", text: w.name }),
        why ? h("span", { class: "s-status-why", text: why }) : null,
        w.state === "stopped" ? h("div", { class: "s-status-hint" }, h("span", { class: "s-status-why", text: "Started and stopped from a terminal:" }), commandBox("agent worker start", "Copy the command to start the worker")) : null);
    } else workerRow = h("div", { class: "ops-row s-status-row" }, pill("neutral", "info", "unknown"), h("span", { class: "s-status-why", text: "Worker status didn’t load." }));

    const servers = ((detail && detail.servers) || []).map((s) => {
      const names = (s.models || []).map((m) => `${m.name}${m.vram_mb ? ` (${gb(m.vram_mb)} GB)` : ""}`);
      return h("div", { class: "ops-row s-status-row" }, s.reachable ? pill("ok", "check", "reachable") : pill("neutral", "x", "unreachable"),
        h("strong", { class: "s-status-name", text: s.name }), h("span", { class: "s-status-url", text: String(s.url || "").replace(/^https?:\/\//, "") }),
        h("span", { class: "s-status-why", text: s.reachable ? (names.length ? `${names.join(", ")} loaded` : "no model loaded") : "not running" }));
    });

    page.replaceChildren(ban, h("div", { class: "s-status-grid" }, gpuCard, cpuCard, hostCard),
      h("section", { class: "s-status-sec" }, sectionHead("Background workers"), workerRow),
      h("section", { class: "s-status-sec" }, sectionHead("Model servers"), servers.length ? servers : h("div", { class: "ops-row s-status-row" }, h("span", { class: "s-status-why", text: "Server status didn’t load." })),
        loaded.length && !servers.length ? h("span", { class: "s-status-why", text: `Loaded: ${loaded.join(", ")}` }) : null),
      h("span", { class: "s-status-stamp mono", text: `Updated ${at.toLocaleTimeString()} · refreshing every 5 s` }));
  }

  async function load() {
    if (first) page.replaceChildren(loadingBox("Reading the machine…"));
    const [a, b, c] = await Promise.allSettled([api.get("/api/status"), api.get("/api/status/detail"), api.get("/api/host/stats")]);
    if (!alive) return;
    if (a.status === "rejected" && b.status === "rejected" && c.status === "rejected") {
      if (first) page.replaceChildren(errorBox(a.reason, load)); return;       // later polls keep the last good picture
    }
    const ok = (r) => (r.status === "fulfilled" ? r.value : null);
    snap = { status: ok(a), detail: ok(b), host: ok(c), at: new Date() };
    first = false; render();
  }

  await load();
  if (alive) timer = setInterval(() => { if (!suspending) load(); }, POLL_MS);
}
