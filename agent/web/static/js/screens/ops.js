// Operations (design S-View "ops") on the view kit, bound to /api/ops/* and the shared settings API:
// governor (state + the real thresholds), background worker (status, start/stop, recent activity), panel, logs, backup and DB sync.
// Backup and sync are report-only because the product has no such commands; their buttons say so instead of pretending.
import { mountView, G, B, T, S, A, F, BR, E, LG } from "../viewkit.js";
import { api } from "../api.js";
import { relTime } from "../dom.js";
import { confirmDialog, toast } from "../ui.js";
import { rowsOf } from "./settings.js";

const pct = (v) => (v == null ? "—" : `${Math.round(v)} %`);
const mb = (b) => (b >= 1073741824 ? `${(b / 1073741824).toFixed(1)} GB` : b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);
const ago = (ts) => (ts ? relTime(ts) : "never");
const fmtLine = (l) => ({ lv: l.lv, t: `${l.ts ? new Date(l.ts * 1000).toLocaleTimeString() + "  " : ""}${l.t}` });
const WORKER_TONE = { running: "ok", paused: "warn", stopped: "neutral", unknown: "neutral" };

export async function mount(root, ctx) {
  const logState = { source: "worker", level: "all", q: "" };
  let logSources = [{ id: "worker", label: "Background worker" }];
  const fetchLogs = async () => {
    const p = new URLSearchParams({ source: logState.source, level: logState.level, q: logState.q, lines: "300" });
    const r = await api.get(`/api/ops/logs?${p}`);
    logSources = r.sources;
    return r.lines.map(fmtLine);
  };

  const tabs = [
    { k: "governor", l: "Resource governor", poll: 5000,
      load: async () => {
        const [g, items] = await Promise.all([api.get("/api/ops/governor"), api.get("/api/settings")]);
        const c = g.caps;
        return [
          g.paused_reason ? B("warn", "Background work is paused", g.paused_reason) : B("ok", "Background work may run", "Nothing is holding it back right now."),
          S([["GPU", pct(g.gpu_percent), c.gpu != null ? `pause above ${c.gpu} %` : ""], ["CPU", pct(g.cpu_percent), c.cpu != null ? `cap ${c.cpu} %` : ""],
            ["RAM", `${g.ram_used_gb}/${g.ram_total_gb} GB`, c.ram_free_gb != null ? `pause below ${c.ram_free_gb} GB free` : ""], ["Game", g.game || "none", c.pause_on_game ? "pauses work" : "ignored"]]),
          G("Thresholds", rowsOf(items, ["pause_on_game", "max_gpu_percent", "max_cpu_percent", "max_ram_gb"], null)),
          B("info", "A pause gate, not a hard cap", "The governor makes background jobs wait between items; the model itself runs inside Ollama and is not limited by these numbers. The GPU and game checks are what protect gaming."),
        ];
      } },

    { k: "worker", l: "Background worker", poll: 5000,
      load: async (vk) => {
        const w = await api.get("/api/ops/worker");
        const control = (action, label, primary) => ({ l: label, primary, disabled: w.state === "unknown" || (action === "start" ? w.state !== "stopped" : w.state === "stopped"), icon: action === "start" ? "play" : "stop",
          onClick: async () => {
            if (action === "stop" && !(await confirmDialog("Stop the background worker?", "Memory distilling, summaries and syncing stop until you start it again.", { confirm: "Stop worker", danger: true }))) return;
            await api.post(`/api/ops/worker/${action}`); toast(action === "start" ? "Worker started" : "Worker stopped"); vk.reload();
          } });
        return [
          ...(w.state === "unknown" ? [B("warn", "systemd did not answer", "The worker state is unknown, so start and stop are unavailable here.")] : []),
          ...(w.state === "paused" ? [B("warn", "Paused by the governor", w.reason || "")] : []),
          S([["State", w.state, ""], ["Starts at login", w.enabled || "unknown", w.unit], ["Doing now", w.activity || "nothing recorded", w.activity_ago_s != null ? `${w.activity_ago_s} s ago` : ""]]),
          T("Worker", ["Unit", "State"], [{ cells: [{ t: w.unit, mono: true }, `${WORKER_TONE[w.state]}:${w.state}`] }], []),
          A([control("start", "Start worker", true), control("stop", "Stop worker")]),
          B("info", "Pausing single jobs is not available yet", "The worker pauses itself by the governor rules (games, GPU, CPU, RAM). There is no job queue to pause by hand."),
          LG("Recent activity", (w.journal || []).map(fmtLine)),
        ];
      } },

    { k: "panel", l: "Panel & status", poll: 5000,
      load: async () => {
        const [p, d] = await Promise.all([api.get("/api/ops/panel"), api.get("/api/status/detail").catch(() => null)]);
        const bars = [["CPU", p.cpu_pct, pct(p.cpu_pct)], ["RAM", (p.ram_used_gb / (p.ram_total_gb || 1)) * 100, `${p.ram_used_gb} / ${p.ram_total_gb} GB`],
          ["Disk (data)", (p.disk_used_gb / (p.disk_total_gb || 1)) * 100, `${p.disk_used_gb} / ${p.disk_total_gb} GB`]];
        if (p.gpu) bars.push(["GPU load", p.gpu.utilization_percent, pct(p.gpu.utilization_percent)], ["VRAM", (p.gpu.vram_used_mb / (p.gpu.vram_total_mb || 1)) * 100, `${(p.gpu.vram_used_mb / 1024).toFixed(1)} / ${(p.gpu.vram_total_mb / 1024).toFixed(1)} GB`]);
        const servers = d ? d.servers : [];
        return [
          S([["Host", p.host, ""], ["Up", p.uptime, ""], ["CPU temp", p.cpu_temp_c == null ? "—" : `${Math.round(p.cpu_temp_c)} °C`, ""], ["Game", p.game || "none", p.paused || ""]]),
          BR("Load", bars),
          S([["Chats", String(p.data.sessions ?? "—"), ""], ["Messages", String(p.data.messages ?? "—"), ""], ["Active facts", String(p.data.facts ?? "—"), ""]], "Data"),
          T("Model servers", ["Server", "Address", "State"], servers.map((s) => ({ cells: [s.name, { t: String(s.url).replace(/^https?:\/\//, ""), mono: true }, s.reachable ? "ok:up" : "neutral:not running"] })), []),
          T("Models in memory (read-only)", ["Model", "Placement"], p.ollama_models.length ? p.ollama_models.map((m) => { const [n, ...w] = m.trim().split(/\s{2,}/); return [{ t: n, mono: true }, w.join(" ") || "—"]; }) : [], []),
        ];
      } },

    { k: "logs", l: "Logs",
      load: async (vk) => {
        let lines = [], err = "";
        try { lines = await fetchLogs(); } catch (e) { err = (e && (e.detail || e.message)) || "Could not read the log."; }
        const live = { ...LG(logSources.find((s) => s.id === logState.source)?.label || "Log", lines), pollMs: 2000, poll: async () => ({ lines: await fetchLogs() }) };
        return [
          F([{ l: "Source", v: logState.source, options: logSources.map((s) => ({ v: s.id, l: s.label })), onInput: (v) => { logState.source = v; vk.reload({ silent: false }); } },
            { l: "Level", v: logState.level, options: [{ v: "all", l: "all" }, { v: "W", l: "warnings and errors" }, { v: "E", l: "errors only" }], onInput: (v) => { logState.level = v; vk.reload({ silent: false }); } },
            { l: "Search", v: logState.q, input: true, placeholder: "Search", onInput: (v) => { logState.q = v; } }]),
          ...(err ? [B("warn", "Could not read the log", err)] : []),
          live,
          ...(!lines.length && !err ? [B("info", "No log lines", "Nothing matches, or the journal has no entries for this unit (journalctl may need your user to be in the systemd-journal group).")] : []),
        ];
      } },

    { k: "backup", l: "Backup",
      load: async () => {
        const b = await api.get("/api/ops/backup");
        return [
          B("warn", "Not available yet", b.reason),
          T("Data on this machine", ["File", "Size", "Changed"], b.files.map((f) => [{ t: f.path, mono: true }, mb(f.bytes), ago(f.modified)]), []),
          G("Backup", [{ label: "Schedule", help: "Not available yet: there is no backup command to schedule.", ctl: "seg:Off|Daily|Weekly", val: "Off", def: "Off", disabled: true },
            { label: "Keep last (n)", help: "Not available yet.", ctl: "nm", val: 7, def: 7, disabled: true }]),
          A([{ l: "Back up now", icon: "download", primary: true, disabled: true }, { l: "Restore…", icon: "undo", disabled: true }]),
        ];
      } },

    { k: "sync", l: "DB sync",
      load: async () => {
        const s = await api.get("/api/ops/sync");
        const rows = [{ cells: [{ t: "This machine (database)", mono: false }, ago(s.database.modified), s.database.exists ? "ok:present" : "neutral:missing"] },
          ...s.scripts.map((x) => ({ cells: [{ t: x.name, mono: true }, ago(x.modified), "info:script found"] }))];
        return [
          B("info", "Read-only", "Shows what this machine has. Comparing two machines needs the sync scripts to be run from a terminal; this screen never starts them."),
          T("Machines", ["Item", "Changed", "State"], rows, []),
          ...(s.scripts.length ? [E("Run a sync", "Open a terminal and run one of the scripts. Pull gets the other machine's data, push sends this one's.", s.scripts[0].path)] : [B("info", "No sync scripts", s.reason)]),
        ];
      } },
  ];
  mountView(root, ctx, { title: "Operations", id: "ops", tabs });
}
