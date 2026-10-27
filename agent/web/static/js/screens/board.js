// Board (design S-Board): blocks the user pinned from answers, as a grid. Reorder by dragging the grip (or the arrow buttons),
// resize with the corner handle (drag, or arrow keys). Layout is saved per item: y = order, w = columns (1-4), h = height in px (0 = auto).
import { h, fmtDateTime } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, confirmDialog } from "../ui.js";
import { renderBlock } from "../components/block.js";

const EXAMPLE = { id: "example", kind: "stats", title: "Example: host numbers", provenance: "computed", source_ids: [],
  spec: { tiles: [{ label: "GPU", value: 41, unit: " °C" }, { label: "VRAM", value: 11.2, unit: " GB" }, { label: "Context", value: 42, unit: " %" }] } };

export async function mount(root, ctx) {
  ctx.css("s-board");
  ctx.shell.setTitle("Board");
  let items = [];
  const page = h("div", { class: "s-board" });
  root.replaceChildren(page);

  const phone = () => matchMedia("(max-width: 820px)").matches;
  const saveError = (e) => toast(e.detail || "Could not save the change");

  async function load() {
    page.replaceChildren(h("div", { class: "l-state", role: "status" }, h("div", { class: "l-skeleton", style: { width: "100%" } }), "Loading the Board…"));
    try { items = await api.get("/api/board/items"); }
    catch (e) {
      page.replaceChildren(h("div", { class: "l-state", role: "alert" }, h("strong", { text: "The Board could not be loaded." }), h("span", { class: "mono", text: e.detail || "Is the server running?" }),
        h("button", { class: "btn btn-secondary", onclick: load }, icon("refresh", 15), "Try again")));
      return;
    }
    draw();
  }

  function draw() {
    if (!items.length) {
      page.replaceChildren(h("div", { class: "s-board-empty" },
        h("h2", { text: "Nothing pinned yet." }),
        h("p", {}, "Pin a chart, table or diagram from any answer with ", h("span", { class: "s-board-inline" }, icon("pin", 16)), " in its toolbar. Blocks with live data can be refreshed here."),
        h("div", { class: "s-board-example", "aria-hidden": "true" }, renderBlock(EXAMPLE, { board: true, onPin: () => {} })),
        h("span", { class: "s-board-note", text: "Example only — it disappears when you pin your first block." })));
      return;
    }
    const live = items.filter((i) => i.refreshable);
    const grid = h("div", { class: "s-board-grid" });
    page.replaceChildren(
      h("div", { class: "s-board-bar" },
        h("span", { class: "s-board-count", text: `${items.length} pinned block${items.length === 1 ? "" : "s"} · drag the grip to reorder, drag a corner to resize` }),
        h("button", { class: "btn btn-secondary", disabled: !live.length, title: live.length ? "" : "None of the pinned blocks has live data", onclick: () => refreshAll(live) }, icon("refresh", 15), "Refresh live blocks")),
      grid);
    items.forEach((it, i) => grid.append(card(it, i, grid)));
  }

  function card(it, i, grid) {
    const fig = renderBlock(it.block, { board: true, boardId: it.id, updated: fmtDateTime(it.updated_at), onPin: () => unpin(it), onRefresh: it.refreshable ? () => refreshOne(it) : null });
    const wrap = h("div", { class: "s-board-item", dataset: { id: it.id } });
    const place = () => { wrap.style.gridColumn = it.w > 1 ? `span ${Math.round(it.w)}` : ""; wrap.style.height = it.h > 0 ? `${Math.round(it.h)}px` : ""; };
    place();

    const grip = h("button", { class: "s-board-ctl-btn", draggable: "true", "aria-label": `Drag ${it.title || "block"} to reorder`, title: "Drag to reorder" }, icon("menu", 16));
    grip.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", String(it.id)); e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setDragImage(wrap, 20, 20); wrap.classList.add("dragging"); });
    grip.addEventListener("dragend", () => wrap.classList.remove("dragging"));
    wrap.addEventListener("dragover", (e) => { e.preventDefault(); wrap.classList.add("over"); });
    wrap.addEventListener("dragleave", () => wrap.classList.remove("over"));
    wrap.addEventListener("drop", (e) => { e.preventDefault(); wrap.classList.remove("over"); const from = items.findIndex((x) => String(x.id) === e.dataTransfer.getData("text/plain")); if (from >= 0) reorder(from, i); });

    const ctl = h("div", { class: "s-board-ctl" }, grip,
      h("button", { class: "s-board-ctl-btn", "aria-label": "Move earlier", disabled: i === 0, onclick: () => reorder(i, i - 1) }, icon("chevu", 16)),
      h("button", { class: "s-board-ctl-btn", "aria-label": "Move later", disabled: i === items.length - 1, onclick: () => reorder(i, i + 1) }, icon("chevd", 16)));
    wrap.append(h("div", { class: "s-board-fig" }, fig), ctl, resizer(it, wrap, grid, place));
    return wrap;
  }

  function resizer(it, wrap, grid, place) {
    const save = async () => { try { Object.assign(it, await api.patch(`/api/board/items/${it.id}`, { w: it.w, h: it.h })); } catch (e) { saveError(e); } };
    const handle = h("button", { class: "s-board-resize", "aria-label": "Resize: arrow keys change width and height", title: "Drag to resize" });
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault(); handle.setPointerCapture(e.pointerId);
      const r = wrap.getBoundingClientRect(), g = getComputedStyle(grid), gap = parseFloat(g.columnGap) || 0;
      const track = parseFloat(g.gridTemplateColumns.split(" ")[0]) || r.width, startW = r.width, startH = r.height;
      const move = (ev) => {
        it.w = Math.min(4, Math.max(1, Math.round((startW + ev.clientX - e.clientX + gap) / (track + gap))));
        it.h = Math.max(140, Math.round(startH + ev.clientY - e.clientY));
        place();
      };
      const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); handle.removeEventListener("pointercancel", up); save(); };
      handle.addEventListener("pointermove", move); handle.addEventListener("pointerup", up); handle.addEventListener("pointercancel", up);
    });
    handle.addEventListener("keydown", (e) => {
      const d = { ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 40], ArrowUp: [0, -40] }[e.key];
      if (!d) return;
      e.preventDefault();
      it.w = Math.min(4, Math.max(1, Math.round(it.w) + d[0]));
      if (d[1]) it.h = Math.max(140, Math.round(it.h || wrap.getBoundingClientRect().height) + d[1]);
      place(); save();
    });
    return handle;
  }

  async function reorder(from, to) {
    if (to < 0 || to >= items.length || from === to) return;
    const before = items.slice();
    items.splice(to, 0, items.splice(from, 1)[0]);
    draw();
    try {
      await Promise.all(items.map((it, idx) => (before[idx] === it ? null : api.patch(`/api/board/items/${it.id}`, { y: idx }))));
      items.forEach((it, idx) => { it.y = idx; });
    } catch (e) { saveError(e); load(); }
  }

  async function unpin(it) {
    if (!(await confirmDialog("Remove from the Board?", `“${it.title || it.block.kind}” stops being pinned. The answer it came from is not changed.`, { confirm: "Remove", danger: true }))) return;
    try { await api.del(`/api/board/items/${it.id}`); items = items.filter((x) => x !== it); draw(); toast("Removed from the Board"); }
    catch (e) { toast(e.detail || "Could not remove it"); }
  }

  async function refreshOne(it) {
    try { Object.assign(it, await api.post(`/api/board/items/${it.id}/refresh`)); draw(); return true; }
    catch (e) { toast(e.detail || "Could not refresh"); return false; }
  }
  async function refreshAll(live) {
    let ok = 0;
    for (const it of live) if (await refreshOne(it)) ok++;
    if (ok) toast(`Refreshed ${ok} block${ok === 1 ? "" : "s"}`);
  }

  await load();
}
