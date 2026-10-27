// Skills (design S-Skills): saved solutions with worked / failed counts, last used, and the "archived by curator" flag.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { toast, confirmDialog } from "../ui.js";

function ago(iso) {
  const t = Date.parse(iso && !/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso + "Z" : iso);
  if (isNaN(t)) return "never";
  const d = Math.floor((Date.now() - t) / 86400000);
  if (d < 1) return "today";
  if (d === 1) return "yesterday";
  if (d < 14) return `${d} days ago`;
  if (d < 60) return `${Math.floor(d / 7)} weeks ago`;
  return `${Math.floor(d / 30)} months ago`;
}

export async function mount(root, ctx) {
  ctx.css("s-skills");
  ctx.shell.setTitle("Skills");
  const page = h("div", { class: "s-skills" });
  root.replaceChildren(page);
  let skills = [];
  const open = new Set();

  async function load() {
    page.replaceChildren(h("div", { class: "l-state", role: "status" }, h("div", { class: "l-skeleton", style: { width: "100%" } }), "Loading skills…"));
    try { skills = await api.get("/api/skills"); }
    catch (e) {
      page.replaceChildren(h("div", { class: "l-state", role: "alert" }, h("strong", { text: "Skills could not be loaded." }), h("span", { class: "mono", text: e.detail || "Is the server running?" }),
        h("button", { class: "btn btn-secondary", onclick: load }, icon("refresh", 15), "Try again")));
      return;
    }
    draw();
  }

  function draw() {
    if (!skills.length) {
      page.replaceChildren(h("div", { class: "s-skills-empty" }, h("h2", { text: "No saved skills." }),
        h("p", { text: "When the assistant solves a problem well, it can keep the steps as a skill and reuse them. They’ll appear here." })));
      return;
    }
    page.replaceChildren(h("div", { class: "s-skills-grid" }, skills.map(card)));
  }

  function card(s) {
    const tags = s.tags || [], archived = tags.includes("archived");
    const reason = (tags.find((t) => String(t).startsWith("reason:")) || "").slice(7);
    const steps = s.solution_steps || [], isOpen = open.has(s.id), panel = `s-skills-steps-${s.id}`;
    return h("article", { class: "card s-skills-card" + (archived ? " archived" : "") },
      h("div", { class: "card-kicker s-skills-kicker", text: `last used ${ago(s.last_used)}` }),
      h("div", { class: "s-skills-head" }, h("h4", { class: "card-title", text: s.title }),
        archived ? h("span", { class: "s-skills-arch", title: reason ? `Reason: ${reason}` : "" }, icon("folder", 12), "archived by curator") : null),
      h("p", { class: "card-body s-skills-problem", text: s.problem_description }),
      h("div", { class: "s-skills-foot" },
        h("span", { class: "s-skills-ok" }, icon("check", 14), `${s.success_count} worked`),
        h("span", { class: "s-skills-bad" }, icon("x", 14), `${s.failure_count} failed`),
        h("span", { class: "s-skills-steps-n", text: `${steps.length} step${steps.length === 1 ? "" : "s"}` }),
        h("span", { class: "s-skills-actions" },
          h("button", { class: "btn btn-ghost", "aria-expanded": String(isOpen), "aria-controls": panel, onclick: () => { isOpen ? open.delete(s.id) : open.add(s.id); draw(); } }, isOpen ? "Hide" : "View"),
          h("button", { class: "btn btn-ghost s-skills-del", "aria-label": `Delete skill ${s.title}`, onclick: () => remove(s) }, "Delete"))),
      isOpen ? h("div", { id: panel },
        steps.length ? h("ol", { class: "s-skills-steps" }, steps.map((t) => h("li", { text: String(t) }))) : h("p", { class: "s-skills-none", text: "This skill has no recorded steps." }),
        (s.tools_used || []).length ? h("p", { class: "s-skills-tools" }, "tools: ", (s.tools_used).map((t) => h("code", { text: t }))) : null) : null);
  }

  async function remove(s) {
    if (!(await confirmDialog("Delete this skill?", `“${s.title}” will be forgotten and can’t be reused. This cannot be undone.`, { confirm: "Delete", danger: true }))) return;
    try { await api.del(`/api/skills/${s.id}`); skills = skills.filter((x) => x.id !== s.id); draw(); toast("Skill deleted"); }
    catch (e) { toast(e.detail || "Could not delete the skill"); }
  }

  await load();
}
