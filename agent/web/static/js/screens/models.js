// Models & GPU (design S-Models): installed models with load state, GPU mode with per-model context estimates,
// chosen context with an over-the-estimate warning, gaming pause and an (estimated) loading state.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, confirmDialog } from "../ui.js";

const fmtSize = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(1) + " GB" : mb + " MB");
const fmtK = (n) => Math.round(n / 1024) + " k";
const fmtTok = (n) => Math.round(n).toLocaleString("en").replace(/,/g, " ");
const errText = (e) => (e && (e.detail || e.message)) || "Something went wrong.";

/** One estimate -> [kind, text]; kind is fit | spill | no | unk (icon + word, never colour alone). */
function cellOf(est, target) {
  if (!est) return ["unk", "unknown"];
  if (!est.fits) return ["no", "doesn’t fit"];
  if (est.ctx < 2048) return ["no", "hardly any room"];
  if (est.approx) return ["unk", `~ ${fmtK(est.ctx)} · rough estimate`];
  if (est.limited_by_model) return ["fit", `${fmtK(est.ctx)} · model max`];
  if (target && est.ctx < target) return ["spill", `${fmtK(est.ctx)} · limited by VRAM`];
  return ["fit", `${fmtK(est.ctx)} · full context`];
}
const CELL = { fit: ["check", "ok"], spill: ["warn", "warn"], no: ["blocked", "danger"], unk: ["shield-q", "neutral"] };

function banner(variant, iconName, strong, text) {
  return h("div", { class: `s-models-ban v-${variant}`, role: variant === "danger" ? "alert" : "status" }, icon(iconName, 18), h("span", {}, h("strong", { text: strong + " " }), text));
}

export async function mount(root, ctx) {
  ctx.css("s-models");
  ctx.shell.setTitle("Models & GPU");

  let loadingModel = null, loadT0 = 0, loadTimer = null, busy = false;
  let gpu = null, draft = { mode: null, ctx: 0 };
  const banners = h("div", { class: "s-models-banners" });
  const loadBox = h("div");
  const modelsBox = h("section", { class: "s-models-sec", "aria-labelledby": "s-models-h" });
  const gpuBox = h("section", { class: "s-models-sec", "aria-labelledby": "s-models-gh" });
  root.replaceChildren(h("div", { class: "l-page s-models" }, banners, loadBox, modelsBox, gpuBox));

  // ------------------------------------------------------------------ status: gaming pause + loaded models
  let status = null, detail = null;
  async function loadStatus() {
    try { status = await api.get("/api/status"); } catch (_) { status = null; }
    banners.replaceChildren(...(status && (status.game_detected || status.paused_reason) ? [banner("warn", "warn", "PC is gaming — background work paused.",
      "Chat still works but may be slower. Background jobs resume when the game closes." + (status.paused_reason && !status.game_detected ? " (" + status.paused_reason + ")" : ""))] : []));
  }
  async function loadDetail() {
    try { detail = { rows: await api.get("/api/models/detail") }; } catch (e) { detail = { error: errText(e) }; }
    drawModels();
  }

  function drawLoading() {
    if (!loadingModel) { loadBox.replaceChildren(); return; }
    const secs = h("span", { class: "mono s-models-secs", text: "0 s" });
    loadBox.replaceChildren(h("div", { class: "l-card s-models-load", role: "status" },
      h("div", { class: "s-models-loadhead" }, h("strong", { text: `Loading ${loadingModel}…` }), secs),
      h("div", { class: "s-models-bar", role: "progressbar", "aria-label": `Loading ${loadingModel}`, "aria-valuetext": "in progress" }, h("i")),
      h("span", { class: "s-models-note", text: "No progress is reported while a model loads — large models can take a minute or two." })));
    clearInterval(loadTimer);
    loadTimer = setInterval(() => { secs.textContent = Math.round((Date.now() - loadT0) / 1000) + " s"; loadStatus(); }, 1000);
  }
  ctx.onCleanup(() => clearInterval(loadTimer));

  async function useModel(name) {
    if (loadingModel) return;
    loadingModel = name; loadT0 = Date.now(); drawLoading(); drawModels();
    try { await api.post("/api/models/select", { model: name }); toast(`${name} is now the chat model`); }
    catch (e) { toast(errText(e)); }
    loadingModel = null; clearInterval(loadTimer); drawLoading();
    await Promise.all([loadStatus(), loadDetail()]);
  }
  async function unloadAll() {
    const ok = await confirmDialog("Unload all models?", "This frees the GPU memory right now. The next message reloads what it needs, which takes a little longer.", { confirm: "Unload all", danger: true });
    if (!ok) return;
    busy = true; drawModels();
    try { const r = await api.post("/api/models/unload"); toast(r.unloaded?.length ? `Unloaded ${r.unloaded.length} model(s)` : "Nothing was loaded"); }
    catch (e) { toast(errText(e)); }
    busy = false;
    await Promise.all([loadStatus(), loadDetail()]);
  }

  function drawModels() {
    const head = h("div", { class: "l-toolbar" }, h("h4", { id: "s-models-h", class: "s-models-h", text: "Models" }),
      h("button", { class: "btn btn-secondary s-models-hit", disabled: busy || !detail?.rows?.some((m) => m.loaded) || null, onclick: unloadAll }, busy ? "Unloading…" : "Unload all"));
    let body;
    if (!detail) body = h("div", { class: "l-list" }, [0, 1, 2].map(() => h("div", { class: "l-skel s-models-skel" })));
    else if (detail.error) body = h("div", { class: "l-state", role: "alert" }, h("strong", { text: "Could not read the model list." }), h("span", { text: detail.error }), h("button", { class: "btn btn-secondary", onclick: loadDetail }, "Try again"));
    else if (!detail.rows.length) body = h("div", { class: "l-state" }, icon("cpu", 22), h("strong", { text: "No models found." }),
      h("span", { text: "Either no model is installed yet, or Ollama is not running. Start Ollama and check again." }), h("button", { class: "btn btn-secondary", onclick: loadDetail }, "Check again"));
    else body = h("div", { class: "l-list" }, detail.rows.map((m) => {
      const isLoading = loadingModel === m.name;
      const stateText = isLoading ? "loading…" : m.loaded ? "loaded in VRAM" : "not loaded";
      return h("div", { class: "s-models-row" },
        h("span", { class: "s-models-dot" + (m.loaded ? " on" : ""), "aria-hidden": "true" }),
        h("strong", { class: "s-models-name", text: m.name }),
        h("span", { class: "s-models-size", text: fmtSize(m.size_mb) }),
        h("span", { class: "s-models-state" + (m.loaded ? " on" : ""), text: stateText }),
        m.current ? h("span", { class: "l-tag s-models-cur" }, icon("check", 12), "current") : null,
        m.current ? null : h("button", { class: "btn btn-secondary s-models-use", disabled: !!loadingModel || null, "aria-label": `Use ${m.name} as the chat model`, onclick: () => useModel(m.name) }, "Use"));
    }));
    modelsBox.replaceChildren(head, body);
  }

  // ------------------------------------------------------------------ GPU mode
  function chatRow() { return gpu.table.find((r) => r.model === (status?.chat_model)) || gpu.table[0] || null; }

  function drawGpu() {
    const head = h("div", {}, h("h4", { id: "s-models-gh", class: "s-models-h", text: "GPU mode" }),
      h("span", { class: "s-models-note", text: "Chosen when the model server starts. It decides how much context a model can use fully on the GPU." }));
    if (gpu === null) { gpuBox.replaceChildren(head, h("div", { class: "l-skel s-models-skel" })); return; }
    if (gpu.error) { gpuBox.replaceChildren(head, h("div", { class: "l-state", role: "alert" }, h("strong", { text: "Could not read the GPU information." }), h("span", { text: gpu.error }), h("button", { class: "btn btn-secondary", onclick: loadGpu }, "Try again"))); return; }
    if (!gpu.options.length) { gpuBox.replaceChildren(head, h("div", { class: "l-state" }, icon("cpu", 22), h("strong", { text: "No dedicated GPU detected." }), h("span", { text: "Models run on the CPU, which is slow. There is nothing to choose here." }))); return; }

    const target = gpu.chosen.ctx || 0;
    const gpuLine = h("span", { class: "s-models-note mono", text: gpu.gpus.map((g) => `${g.name} (${fmtSize(g.used_mb)} used of ${fmtSize(g.total_mb)})`).join(" · ") });

    const segOpts = gpu.options.map((o) => {
      const input = h("input", { type: "radio", name: "s-models-gpu", value: o.mode, checked: draft.mode === o.mode || null,
        onchange: () => { draft.mode = o.mode; refreshDerived(); } });
      return h("label", { class: "seg-opt s-models-hit" }, input, `${o.label} · ${o.usable_gb} GB`);
    });
    const seg = h("div", { class: "seg s-models-seg", role: "radiogroup", "aria-label": "GPU mode" }, segOpts);

    const table = h("table", { class: "s-models-table" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col", text: "Model" }), gpu.options.map((o) => h("th", { scope: "col", text: o.mode === "single" ? "One GPU" : o.mode === "dual" ? "Both GPUs" : o.label })))),
      h("tbody", {}, gpu.table.map((r) => h("tr", { class: r === chatRow() ? "chat" : "" }, h("td", {}, h("span", { class: "mono", text: r.model })),
        gpu.options.map((o, i) => {
          const [kind, text] = cellOf(r.estimates[i], target);
          const [ic, tone] = CELL[kind];
          return h("td", { dataset: { mode: o.mode } }, h("span", { class: `s-models-cell v-${tone}` }, icon(ic, 14), text));
        })))));

    const slider = h("input", { id: "s-models-ctx", type: "range", min: 8, max: Math.max(160, Math.ceil(draft.ctx / 1024)), step: 1, value: Math.round(draft.ctx / 1024), class: "s-models-range",
      "aria-describedby": "s-models-ctx-help", oninput: () => { draft.ctx = Number(slider.value) * 1024; refreshDerived(); } });
    const ctxLabel = h("span", { class: "mono s-models-ctxlabel" });
    const help = h("span", { id: "s-models-ctx-help", class: "s-models-note" });
    const kvVal = h("input", { id: "s-models-kv", class: "input s-models-hit mono", readonly: true, "aria-describedby": "s-models-kv-help" });
    const warnBox = h("div");
    const applyBtn = h("button", { class: "btn btn-primary s-models-hit", onclick: apply }, "Apply GPU mode");
    const status_ = h("span", { class: "s-models-note" });

    function refreshDerived() {
      for (const td of table.querySelectorAll("td[data-mode]")) td.classList.toggle("sel", td.dataset.mode === draft.mode && td.parentElement.classList.contains("chat"));
      const optIdx = gpu.options.findIndex((o) => o.mode === draft.mode);
      const opt = gpu.options[optIdx], row = chatRow(), est = row && optIdx >= 0 ? row.estimates[optIdx] : null;
      ctxLabel.textContent = fmtTok(draft.ctx);
      kvVal.value = opt ? opt.kv_type : "";
      help.textContent = est && est.fits ? `Estimate for ${opt.mode === "dual" ? "both GPUs" : "one GPU"}: up to ${fmtTok(est.ctx)} tokens fully on the GPU${est.limited_by_model ? " (the model’s own maximum)" : ""}.`
        : est ? `${row.model} does not fit completely in this mode.` : "No estimate for this model.";
      if (est && est.fits && est.ctx > 0 && draft.ctx > est.ctx) {
        const over = draft.ctx - est.ctx;
        warnBox.replaceChildren(banner("danger", "danger", `Above the estimate by ${fmtTok(over)} tokens (${Math.round((over / est.ctx) * 100)} %).`,
          "Some layers will spill to the CPU and generation gets much slower. Lower the context, or keep it and accept the slowdown."));
      } else warnBox.replaceChildren();
      status_.textContent = gpu.chosen.mode ? `In effect now: ${gpu.chosen.mode === "dual" ? "both GPUs" : "one GPU"}${gpu.chosen.ctx ? ", " + fmtTok(gpu.chosen.ctx) + " tokens" : ""}.` : "No mode chosen yet — the CLI asks on start.";
    }

    gpuBox.replaceChildren(head, gpuLine, seg, h("div", { class: "s-models-scroll" }, table),
      h("div", { class: "s-models-grid" },
        h("div", { class: "field" }, h("label", { for: "s-models-ctx", text: "Chat context size" }), h("div", { class: "s-models-sliderrow" }, slider, ctxLabel), help),
        h("div", { class: "field" }, h("label", { for: "s-models-kv", text: "KV-cache type" }), kvVal, h("span", { id: "s-models-kv-help", class: "s-models-note", text: "Set by the GPU mode; not editable here." }))),
      warnBox, h("div", { class: "l-toolbar" }, applyBtn, status_));
    refreshDerived();

    async function apply() {
      const changing = gpu.chosen.mode && gpu.chosen.mode !== draft.mode;
      if (changing && !(await confirmDialog("Switch GPU mode?", "Models already loaded stay on their current GPUs until they are unloaded and loaded again.", { confirm: "Switch mode" }))) return;
      applyBtn.disabled = true;
      try { await api.put("/api/gpu", { mode: draft.mode, ctx: draft.ctx }); toast("GPU mode applied"); await loadGpu(true); }
      catch (e) { toast(errText(e)); applyBtn.disabled = false; }
    }
  }

  async function loadGpu(keepDraft = false) {
    try {
      gpu = await api.get("/api/gpu");
      if (!keepDraft || !draft.mode) {
        const row = chatRow();
        const first = gpu.options[0];
        draft.mode = gpu.chosen.mode || gpu.last_mode || (first && first.mode) || null;
        const oi = Math.max(0, gpu.options.findIndex((o) => o.mode === draft.mode));
        const est = row && row.estimates[oi];
        draft.ctx = gpu.chosen.ctx || (est && est.ctx) || 32768;
      }
    } catch (e) { gpu = { error: errText(e) }; }
    drawGpu();
  }

  drawModels(); drawGpu();
  await loadStatus();
  await Promise.all([loadDetail(), loadGpu()]);
  const poll = setInterval(() => { if (!document.hidden && !loadingModel) { loadStatus(); loadDetail(); } }, 6000);
  ctx.onCleanup(() => clearInterval(poll));
}
