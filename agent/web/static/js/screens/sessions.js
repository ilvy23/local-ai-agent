// Sessions & chat tools: manage sessions, per-turn controls reference, slash commands reference, undo, notes & plan,
// attachments & media - live where the product has the capability, honestly disabled where it does not (yet).
import { h, fmtDateTime, relTime } from "../dom.js";
import { api, getToken } from "../api.js";
import { confirmDialog, dialog, download, toast } from "../ui.js";
import { mountView, B, D, G, T, C, F } from "../viewkit.js";
import area from "../views-data/sessions.js";

const manageTab = area.tabs.find((t) => t.k === "manage");
const slashTab = area.tabs.find((t) => t.k === "slash");

async function exportSession(id, format, title) {
  const res = await fetch(`/api/sessions/${id}/export?format=${format}`, { headers: getToken() ? { Authorization: "Bearer " + getToken() } : {} });
  if (!res.ok) return toast("Could not export");
  const blob = await res.blob();
  download(`${(title || "session").replace(/[^\w -]/g, "_")}.${format === "json" ? "json" : "md"}`, blob);
}

export async function mount(root, ctx) {
  let search = "", sort = "recent", pinnedFirst = true;

  async function manageLoad(vk) {
    const sessions = await api.get("/api/sessions?limit=500");
    let rows = sessions.filter((s) => (s.title || "New chat").toLowerCase().includes(search.toLowerCase()));
    rows.sort((a, b) => sort === "oldest" ? a.id - b.id : sort === "messages" ? b.message_count - a.message_count : b.id - a.id);
    if (pinnedFirst) rows.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0));

    const items = rows.map((s) => ({
      t: s.title || "New chat",
      sub: `${s.message_count} message${s.message_count === 1 ? "" : "s"} · ${s.started_at ? relTime(s.started_at) : "—"}`,
      tag: s.pinned ? "neutral:pinned" : "",
      _row: s,
    }));

    async function detailFor(item) {
      const s = item._row;
      return {
        title: s.title || "New chat",
        rows: [["Started", s.started_at ? fmtDateTime(s.started_at) : "unknown"], ["Messages", String(s.message_count)],
                ["Pinned", s.pinned ? "yes" : "no"]],
        acts: [
          { l: "Resume", primary: true, onClick: () => ctx.navigate("#/chat/" + s.id) },
          { l: "Rename", onClick: async () => {
            const r = await dialog({ title: "Rename session", body: renameField(s.title || ""), actions: [{ id: "cancel", label: "Cancel" }, { id: "ok", label: "Rename", primary: true }] });
            if (r !== "ok") return;
            const title = document.getElementById("vk-rename-input")?.value?.trim();
            if (!title) return;
            await api.patch(`/api/sessions/${s.id}`, { title });
            await ctx.shell.loadSessions(); vk.reload();
          } },
          { l: s.pinned ? "Unpin" : "Pin", onClick: async () => { await api.patch(`/api/sessions/${s.id}`, { pinned: !s.pinned }); vk.reload(); } },
          { l: "Export Markdown", onClick: () => exportSession(s.id, "markdown", s.title) },
          { l: "Export JSON", onClick: () => exportSession(s.id, "json", s.title) },
          { l: "Fork from here", onClick: async () => {
            const msgs = await api.get(`/api/sessions/${s.id}/messages`);
            if (!msgs.length) return toast("Nothing to fork yet");
            const forked = await api.post(`/api/sessions/${s.id}/fork`, { message_id: msgs[msgs.length - 1].id });
            await ctx.shell.loadSessions(); vk.reload();
            toast("Forked"); ctx.navigate("#/chat/" + forked.id);
          } },
          { l: "Delete…", danger: true, onClick: async () => {
            if (!(await confirmDialog("Delete this chat?", "The messages are removed. Facts learned from it stay in memory.", { confirm: "Delete", danger: true }))) return;
            await api.del(`/api/sessions/${s.id}`);
            await ctx.shell.loadSessions(); vk.reload();
          } },
        ],
      };
    }

    return [
      F([
        { l: "Search sessions", v: search, i: "search", input: true, placeholder: "Search sessions", onInput: (v) => { search = v; vk.reload(); } },
        { l: "Sort", v: sort, i: "list", options: [{ v: "recent", l: "Recent" }, { v: "oldest", l: "Oldest" }, { v: "messages", l: "Most messages" }], onInput: (v) => { sort = v; vk.reload(); } },
        { l: "Pinned first", v: pinnedFirst ? "On" : "Off", i: "pin", onClick: () => { pinnedFirst = !pinnedFirst; vk.reload(); } },
      ]),
      { k: "ld", title: "All sessions", items, detailFor, emptyText: "No sessions match." },
      { ...D("Delete all sessions", "Removes every chat and message. Memory facts stay. This cannot be undone.", "delete", "Delete all sessions"),
        onConfirm: async () => { await api.del("/api/sessions"); await ctx.shell.loadSessions(); vk.reload(); toast("All sessions deleted"); } },
    ];
  }

  function renameField(value) {
    const inp = h("input", { id: "vk-rename-input", class: "input", value, "aria-label": "Title" });
    setTimeout(() => inp.focus(), 0);
    return inp;
  }

  async function undoLoad() {
    return [B("info", "Undo isn't tracked for web chat yet",
      "The file-change undo stack only exists in the terminal chat right now (agent/undo.py's UndoStack lives in that process). " +
      "A web session can't restore or diff file snapshots from here yet.")];
  }

  async function notesLoad(vk) {
    const sessionIds = (await api.get("/api/sessions?limit=1")).map((s) => s.id);
    const currentId = sessionIds[0];
    const sections = [];
    if (currentId != null) {
      const n = await api.get(`/api/sessions/${currentId}/notes`);
      sections.push(G(`Session notes (most recent chat, #${currentId})`, [
        { label: "Notes for this chat", help: "Kept with the session; the assistant can read them.", ctl: "ta", val: n.notes || "", def: "(empty)",
          onChange: async (v) => { await api.put(`/api/sessions/${currentId}/notes`, { notes: v }); } },
      ]));
    } else {
      sections.push(B("info", "No chat yet", "Start a chat to give it notes."));
    }
    sections.push(B("warn", "Plan isn't available here yet",
      "The acceptance-list checker (/plan) only lives in the terminal session's memory right now - it is never persisted, so the web GUI has nothing to show."));
    const projects = await api.get("/api/projects");
    if (projects.length) {
      const proj = projects[0];
      const notes = await api.get(`/api/projects/notes?key=${encodeURIComponent(proj.key)}`);
      sections.push(T(`Project notes - ${proj.name}`, ["Note"], notes.map((n) => [n.note]),
        [{ l: "Clear all", danger: true, onClick: async () => { await api.del(`/api/projects/notes?key=${encodeURIComponent(proj.key)}`); vk.reload(); } }]));
    } else {
      sections.push(B("info", "No project notes yet", "Notes appear once the code agent has worked in a folder."));
    }
    return sections;
  }

  async function filesLoad() {
    const sessionIds = (await api.get("/api/sessions?limit=1")).map((s) => s.id);
    const currentId = sessionIds[0];
    if (currentId == null) return [B("info", "No chat yet", "Attachments appear once you send a message with a file.")];
    const msgs = await api.get(`/api/sessions/${currentId}/messages`);
    const cards = msgs.filter((m) => m.role === "file_card").map((m) => { try { return JSON.parse(m.content); } catch (_) { return null; } }).filter(Boolean);
    return [
      B("info", "Attachments you send are read in the chat itself",
        "Files you attach to a message show up in that chat's history (where you sent them) - they aren't tracked separately on this screen."),
      cards.length ? C("Sent back by the assistant (most recent chat)", cards.map((c) => ({ t: c.filename, s: `${c.kind || "file"} · ${c.size || ""}`, img: 1 })))
        : B("info", "Nothing sent back yet", "Files the assistant generates or sends (images, exports) will appear here."),
    ];
  }

  mountView(root, ctx, {
    title: area.title,
    tabs: [
      { k: "manage", l: "Sessions", load: manageLoad },
      { k: "turn", l: "Per-turn controls", s: [
        T("Controls around one message", ["Control", "Where", "State"], [
          ["Force web search", "Composer /web toggle", "warn:button only: the server does not act on it yet"],
          ["Regenerate", "Under the last answer", "ok:available"],
          ["Edit last message", "Not in the web composer", "neutral:not available yet"],
          ["Stop", "Send button becomes Stop while running", "ok:available"],
          ["Model for one message", "Not in the web composer (model is chosen in Models)", "neutral:not available yet"],
          ["Think harder / quick answer", "Needs a model capability flag", "neutral:not available yet"],
        ]),
      ] },
      { k: "slash", l: "Slash commands", s: [...slashTab.s, B("info", "Not in the web chat yet", "/plan, /project and /undo only work in the terminal chat; the web composer says so when you type them.")], needs: [] },
      { k: "undo", l: "Undo", load: undoLoad },
      { k: "notes", l: "Notes & plan", load: notesLoad },
      { k: "files", l: "Attachments & media", load: filesLoad },
    ],
  });
}
