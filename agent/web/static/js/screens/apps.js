// Apps: images, files the assistant shared, and the prompt studio's saved chats / refine loop / Max trace. Optional features
// (see features.js) may add further tabs. Routes: routes/images.py, routes/apps.py.
import { mountView, G, S, E, B, A, T, BR } from "../viewkit.js";
import { api } from "../api.js";
import { toast, copyText, confirmDialog } from "../ui.js";
import { h, relTime } from "../dom.js";
import { albumCard } from "../components/mediaplayer.js";
import { featureTabs } from "../features.js";

const mb = (b) => (b == null ? "—" : b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " kB");
const secs = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
const NA = (why) => ({ disabled: true, help: why });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Start a prompt job and wait for it; throws the server's message on failure. */
async function runPrompt(body, onTick) {
  const { job } = await api.post("/api/prompt/jobs", body);
  for (;;) {
    await sleep(1200);
    const j = await api.get(`/api/prompt/jobs/${job}`);
    if (onTick) onTick(j);
    if (j.status === "done") return j.result;
    if (j.status === "failed") throw new Error(j.error || "The model call failed");
    if (j.status === "stopped") throw new Error("Stopped");
  }
}

export async function mount(root, ctx) {
  let vk = null;
  const st = { genPrompt: "", genSize: "1024", genSession: null, resume: null, raw: "", feedback: "", refined: "", maxText: "", trace: null, busy: "" };

  // ------------------------------------------------------------------------------------------------ images
  const images = async () => {
    const [info, gal, q] = await Promise.all([api.get("/api/images/info"), api.get("/api/images/gallery"), api.get("/api/images/queue")]);
    const chats = (ctx.shell.sessions || []).slice(0, 12);
    if (st.genSession == null && chats[0]) st.genSession = chats[0].id;
    const running = q.filter((j) => j.status === "running");
    return [
      S([["Model", info.installed ? "installed" : "missing", info.model], ["Typical time", secs(info.eta_s), "median of recent runs"], ["Now", info.busy ? "busy" : "free", "one at a time"]]),
      !info.installed ? B("warn", "Image model not installed", `The local image model is not at the expected place, so nothing can be generated yet (${info.output_dir}).`) : null,
      gal.length ? { k: "cards", title: "Generated", items: gal.map((g) => ({ t: g.title, s: `${g.exists ? mb(g.size) : "file gone"} · ${relTime(g.created_at)}`, src: g.exists ? g.url : undefined, tag: g.exists ? undefined : "warn:missing",
        onClick: () => g.exists && window.open(g.url, "_blank", "noopener") })) } : E("No images yet", "Pictures made in chat or below show up here. Seed, steps and time per picture are not recorded by the generator.", ""),
      ...q.map((j) => ({ k: "job", title: j.prompt.slice(0, 80), sub: j.status === "failed" ? (j.error || "failed") : `Estimate from the median of recent runs; not a step count`, pct: j.pct, st: j.status === "running" ? "running" : j.status === "done" ? "done" : "failed",
        el: secs(j.elapsed_s), eta: j.status === "running" ? secs(Math.max(0, j.eta_s - j.elapsed_s)) + " left" : "—",
        acts: [{ l: "Cancel", disabled: true, icon: "x" }], poll: running.some((r) => r.id === j.id) ? async () => { const n = (await api.get("/api/images/queue")).find((x) => x.id === j.id);
          if (n && n.status !== "running") queueMicrotask(() => vk.reload()); return n ? { pct: n.pct, el: secs(n.elapsed_s), st: n.status === "running" ? "running" : n.status === "done" ? "done" : "failed" } : null; } : undefined, pollMs: 4000 })),
      q.length ? B("info", "Cancel is not available yet", "A running picture cannot be stopped from here; it finishes or fails on its own.") : null,
      G("Make a picture", [
        { label: "Prompt", ctl: "ta", val: st.genPrompt, def: "", onChange: (v) => { st.genPrompt = v; } },
        { label: "Size", help: "Pixels per side.", ctl: "seg", opts: ["768", "1024"], val: st.genSize, def: "1024", onChange: (v) => { st.genSize = v; } },
        { label: "Deliver into chat", help: "The finished picture appears in that chat, like one asked for in conversation.", ctl: "sel", opts: chats.map((c) => ({ v: String(c.id), l: c.title || "Untitled" })), val: String(st.genSession ?? ""), def: "", onChange: (v) => { st.genSession = Number(v); } },
      ]),
      A([{ l: "Generate", icon: "play", primary: true, disabled: !info.installed || info.busy || !chats.length, onClick: async () => {
        if (!st.genPrompt.trim()) return toast("Write a prompt first");
        await api.post("/api/images/generate", { prompt: st.genPrompt, session_id: st.genSession, size: Number(st.genSize) }); toast("Started; it takes several minutes"); st.genPrompt = ""; vk.reload(); } }]),
      G("Settings", [
        { label: "Steps", ctl: "nm", val: "", def: "", ...NA("Not available yet: the generator's script chooses the steps.") },
        { label: "Seed", ctl: "nm", val: "", def: "", ...NA("Not available yet: pictures are not seeded or reproducible from here.") },
      ]),
    ].filter(Boolean);
  };

  // ------------------------------------------------------------------------------------------------ shared files
  const files = async () => {
    const rows = await api.get("/api/apps/shared-files");
    const openRow = (f) => (f.kind === "album" ? ctx.shell.openInspector(h("div", { class: "l-page" }, albumCard({ ...f, album_kind: f.album_kind, size: mb(f.size) }))) : window.open(f.url, "_blank", "noopener"));
    return [T("Files the assistant shared", ["File", "Type", "Shared", "Status", ""], rows.map((f) => ({ cells: [{ t: f.kind === "album" ? `${f.name} (${f.count} ${f.album_kind === "image" ? "photos" : f.album_kind === "video" ? "videos" : "tracks"})` : f.name, mono: true }, f.kind === "album" ? "album" : f.kind, relTime(f.created_at),
      f.exists ? "ok:available" : "warn:file gone", f.exists ? { t: "Open", btn: true, onClick: () => openRow(f) } : "—"] })), []),
      B("info", "Links do not expire yet", "A shared link works for as long as the file is on disk; there is no expiry or revoke in the product. Anyone holding a link can open that one file (the link is a 256-bit random token); the list above is only visible after sign-in."),
    ];
  };

  // ------------------------------------------------------------------------------------------------ prompt studio
  const busyOp = async (label, fn) => { st.busy = label; try { return await fn(); } finally { st.busy = ""; } };
  const prompt = async () => {
    const rows = await api.get("/api/prompt/sessions");
    const sel = st.resume && rows.some((r) => r.id === st.resume.id) ? st.resume : null;
    const out = [
      { k: "ld", title: "Saved prompt chats", emptyText: "None yet. Start one below.", items: rows.map((r) => ({ t: r.title || "Untitled", sub: `${r.mode} · ${r.created}` })),
        detailFor: async (it, i) => { const d = await api.get(`/api/prompt/sessions/${rows[i].id}`);
          return { title: it.t, rows: [["Mode", d.mode], ["Messages", String(d.messages.length)], ["Last reply", (rows[i].last || "").slice(0, 140) || "—"]], acts: [
            { l: "Resume", primary: true, onClick: () => { st.resume = { id: d.id, mode: d.mode, messages: d.messages }; vk.reload(); } },
            { l: "Copy last reply", onClick: () => copyText(rows[i].last || "", "Copied") },
            { l: "Delete", danger: true, onClick: async () => { if (await confirmDialog("Delete this saved chat?", "It is removed from your saved prompt chats.", { confirm: "Delete", danger: true })) { await api.del(`/api/prompt/sessions/${d.id}`); if (st.resume && st.resume.id === d.id) st.resume = null; vk.reload(); } } }] }; } },
      { k: "chat", title: sel ? `Chat: ${rows.find((r) => r.id === sel.id).title}` : "New prompt chat", msgs: sel ? sel.messages.map((m) => [m.role === "user" ? "you" : "bot", m.content]) : [],
        onSend: async (t) => { const r = await runPrompt({ op: "chat", mode: sel ? sel.mode : "improve", sid: sel ? sel.id : null, message: t }); st.resume = { id: r.sid, mode: sel ? sel.mode : "improve", messages: [] }; setTimeout(() => vk.reload(), 0); return r.text; } },
      G("Refine loop", [
        { label: "Prompt to refine", ctl: "ta", val: st.raw, def: "", onChange: (v) => { st.raw = v; } },
        { label: "Your feedback", help: "Say what to change; it revises.", ctl: "ta", val: st.feedback, def: "", onChange: (v) => { st.feedback = v; } }]),
      A([{ l: "Revise", icon: "refresh", primary: true, onClick: async () => { const r = await runPrompt({ op: "refine", current: st.refined || st.raw, feedback: st.feedback }); st.refined = r.text; vk.reload(); } },
         { l: "Start over", onClick: () => { st.refined = ""; vk.reload(); } }]),
      st.refined ? { k: "text", title: "Revised prompt", text: st.refined } : null,
      st.refined ? A([{ l: "Copy", icon: "copy", onClick: () => copyText(st.refined, "Copied") }]) : null,
      G("Max anonymise", [{ label: "Text to anonymise", help: "Five passes on your own model (distil, keep, strip, rebuild, verify). Slow: a few minutes.", ctl: "ta", val: st.maxText, def: "", onChange: (v) => { st.maxText = v; } }]),
      A([{ l: "Run Max anonymise", icon: "shield-half", primary: true, onClick: async () => { const r = await runPrompt({ op: "anon", text: st.maxText, strength: "max" }); st.trace = r; vk.reload(); } }]),
      st.trace ? { k: "text", title: "Anonymised result", text: st.trace.text } : null,
      st.trace ? T("Max anonymise: trace", ["Pass", "Result"], st.trace.trace.map((p) => [{ t: p.pass, mono: true }, p.text]), []) : null,
    ];
    return out.filter(Boolean);
  };

  const extra = await featureTabs("apps", ctx, () => vk);
  vk = mountView(root, ctx, { title: "Apps", id: "apps", tabs: [
    ...extra,
    { k: "images", l: "Images", load: images },
    { k: "files", l: "Shared files", load: files },
    { k: "prompt", l: "Prompt studio", load: prompt },
  ] });
}
