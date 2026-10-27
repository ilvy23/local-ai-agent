// Projects & folder access (design S-Projects): what the assistant may do in each folder without asking, level by level,
// with the blast radius spelled out. Home and system folders are refused by the server and never offered.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, dialog, confirmDialog } from "../ui.js";
import { pill, loadingBox, errorBox, emptyBox } from "../components/ops-ui.js";

const LEVELS = [["ask", "Ask"], ["medium", "Medium"], ["auto", "Auto"], ["greenlist", "Always allow"]];
// per level: [edits files free, runs commands free]
const FREE = { ask: [0, 0], medium: [1, 0], auto: [1, 1], greenlist: [1, 1] };
const BLAST = {
  auto: "Auto lets the assistant edit files and run scoped commands in this folder without asking, until the server restarts.",
  greenlist: "Always allow does the same as Auto, and it is remembered forever, in every future session, until you revoke it.",
};

export async function mount(root, ctx) {
  ctx.css("ops-ui"); ctx.css("s-projects");
  const page = h("div", { class: "l-page" });
  root.replaceChildren(page);
  const asked = new Set();                      // folders set back to "ask" during this visit stay listed until you leave
  let data = null;

  async function load() {
    page.replaceChildren(loadingBox("Loading folders…"));
    try {
      const [access, projects, sessions] = await Promise.all([api.get("/api/access"), api.get("/api/projects"), api.get("/api/code/sessions").catch(() => [])]);
      data = { access, projects, sessions };
    } catch (e) { page.replaceChildren(errorBox(e, load)); return; }
    render();
  }

  function folders() {
    const map = new Map();
    const get = (path) => { if (!map.has(path)) map.set(path, { path, level: "ask", forever: false, notes: 0, active: false }); return map.get(path); };
    for (const s of data.access.session || []) get(s.path).level = s.level;
    for (const p of data.access.greenlist || []) { const f = get(p); f.level = "greenlist"; f.forever = true; }
    for (const p of data.projects || []) get(p.key).notes = p.notes;
    for (const p of asked) get(p);
    for (const s of data.sessions || []) if (s.state === "in progress" && map.has(s.root)) map.get(s.root).active = true;
    return [...map.values()];
  }

  async function put(path, level) { return api.put("/api/access", { path, level }); }

  async function setLevel(f, level) {
    if (level === f.level) return;
    if (BLAST[level] && !(await confirmDialog(`Give “${LEVELS.find((l) => l[0] === level)[1]}” access?`, `${f.path}\n\n${BLAST[level]}`, { confirm: "Allow" }))) { render(); return; }
    try {
      if (f.forever) await put(f.path, "ungreenlist");        // leaving "always allow" first drops the saved entry
      await put(f.path, level);
      level === "ask" ? asked.add(f.path) : asked.delete(f.path);
      toast(level === "ask" ? "Back to asking in this folder" : "Access updated");
    } catch (e) { toast(e.detail || "Could not change access", 4200); }
    await refresh();
  }

  async function revoke(f) {
    if (!(await confirmDialog("Revoke access?", `${f.path}\n\nThe assistant goes back to asking for every edit and command here.`, { confirm: "Revoke access", danger: true }))) return;
    try {
      if (f.forever) await put(f.path, "ungreenlist");
      await put(f.path, "ask");
      asked.delete(f.path);
      toast("Access revoked");
    } catch (e) { toast(e.detail || "Could not revoke access", 4200); }
    await refresh();
  }

  async function clearNotes(f) {
    if (!(await confirmDialog("Forget project notes?", `The assistant forgets what it noted about ${f.path} (${f.notes} note${f.notes === 1 ? "" : "s"}). This cannot be undone.`, { confirm: "Forget notes", danger: true }))) return;
    try { await api.del(`/api/projects/notes?key=${encodeURIComponent(f.path)}`); toast("Notes forgotten"); }
    catch (e) { toast(e.detail || "Could not forget the notes", 4200); }
    await refresh();
  }

  async function refresh() {
    try {
      const [access, projects, sessions] = await Promise.all([api.get("/api/access"), api.get("/api/projects"), api.get("/api/code/sessions").catch(() => [])]);
      data = { access, projects, sessions };
    } catch (e) { toast(e.detail || "Could not refresh"); }
    render();
  }

  async function addFolder() {
    let path = "", level = "medium";
    for (;;) {
      const input = h("input", { class: "input", id: "s-projects-add", value: path, placeholder: "/home/you/code/project", autocomplete: "off", spellcheck: "false", style: { fontFamily: "var(--font-mono)" } });
      const seg = h("div", { class: "seg", role: "radiogroup", "aria-label": "Access level", style: { flexWrap: "wrap" } },
        ["medium", "auto", "greenlist"].map((l) => h("label", { class: "seg-opt", style: { minHeight: "44px" } },
          h("input", { type: "radio", name: "s-projects-add-level", value: l, checked: l === level, onchange: () => { level = l; } }), LEVELS.find((x) => x[0] === l)[1])));
      const r = await dialog({ title: "Allow a folder", actions: [{ id: "no", label: "Cancel" }, { id: "yes", label: "Allow folder", primary: true }],
        body: h("div", { class: "s-projects-form" },
          h("div", { class: "field" }, h("label", { for: "s-projects-add", text: "Absolute path of a project folder" }), input),
          seg, h("span", { class: "mono", text: "Home and system folders are refused; pick a project folder inside them." })) });
      path = input.value.trim();
      if (r !== "yes") return;
      try { await put(path, level); toast("Folder allowed"); await refresh(); return; }
      catch (e) { toast(e.detail || "That folder can’t be allowed", 4600); }
    }
  }

  function capCell(label, text, free) {
    return h("div", { class: "s-projects-cap " + (free ? "v-ok" : "v-warn") }, icon(free ? "check" : "lock", 15),
      h("span", {}, h("strong", { text: label }), h("br"), text));
  }

  function noteBox(f) {
    const list = h("ul", {});
    let loaded = false;
    const box = h("details", { class: "s-projects-notes" },
      h("summary", {}, h("strong", { text: "Project notes" }), ` · ${f.notes} note${f.notes === 1 ? "" : "s"}`), list);
    box.addEventListener("toggle", async () => {
      if (!box.open || loaded) return;
      loaded = true;
      list.replaceChildren(h("li", { text: "Loading…" }));
      try {
        const notes = await api.get(`/api/projects/notes?key=${encodeURIComponent(f.path)}`);
        list.replaceChildren(...(notes.length ? notes.map((n) => h("li", { text: n.note })) : [h("li", { text: "No notes." })]));
      } catch (e) { loaded = false; list.replaceChildren(h("li", { text: e.detail || "Could not load the notes." })); }
    });
    return box;
  }

  function card(f, i) {
    const free = FREE[f.level];
    const standing = f.level !== "ask";
    return h("article", { class: "l-card s-projects-card", "aria-label": f.path },
      h("div", { class: "s-projects-head" }, icon("folder", 18), h("strong", { class: "s-projects-path", text: f.path }),
        f.active ? pill("info", "check", "agent working here") : null, f.forever ? pill("warn", "lock", "remembered forever") : null),
      h("div", { class: "seg", role: "radiogroup", "aria-label": `Access level for ${f.path}` },
        LEVELS.map(([id, label]) => h("label", { class: "seg-opt" },
          h("input", { type: "radio", name: `lvl-${i}`, value: id, checked: id === f.level, onchange: () => setLevel(f, id) }), label))),
      h("div", { class: "s-projects-caps" },
        capCell("Edits files", free[0] ? "free, no prompt" : "asks you each time", free[0]),
        capCell("Runs commands", free[1] ? "only scoped commands, free" : "asks you each time", free[1]),
        capCell("Outside this folder", "always asks", false)),
      f.notes ? noteBox(f) : null,
      h("div", { class: "s-projects-actions" },
        standing ? h("button", { class: "btn btn-secondary", onclick: () => revoke(f) }, "Revoke access") : null,
        f.notes ? h("button", { class: "btn btn-ghost", onclick: () => clearNotes(f) }, "Clear notes") : null));
  }

  function render() {
    const list = folders();
    page.replaceChildren(
      h("div", { class: "s-projects-intro" },
        h("p", {}, "What the assistant may do in each folder ", h("strong", { text: "without asking" }), ". Anything not listed always asks."),
        h("button", { class: "btn btn-primary", style: { minHeight: "44px" }, onclick: addFolder }, icon("plus", 15), "Allow a folder")),
      list.length
        ? h("div", { class: "s-projects-list" }, list.map(card))
        : emptyBox("folder", "No folders allowed yet", "Every edit and command asks first. Allow a project folder to let the assistant work in it on its own.",
          h("button", { class: "btn btn-primary", style: { minHeight: "44px" }, onclick: addFolder }, icon("plus", 15), "Allow a folder")),
      h("div", { class: "s-projects-list" },
        h("div", { class: "l-card s-projects-dim" },
          h("div", { class: "s-projects-head" }, icon("lock", 18), h("strong", { class: "s-projects-path", text: "~  ·  /etc  ·  /" }), pill("block", "blocked", "not offered", "s-projects-block")),
          h("span", { style: { fontSize: "var(--fs-sm)", color: "var(--color-text-2)" }, text: "Home and system folders contain too much to hand over at once — passwords, keys, other projects. Allow a project folder inside them instead." }))),
      h("p", { class: "s-projects-note-small", text: "Medium and Auto last until the server restarts. Always allow is saved and applies to every future session." }));
  }

  await load();
}
