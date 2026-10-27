// Approval card (design L-Approval). CAUTION: Approve / Deny. DANGEROUS: type "yes" to enable Approve and run.
// BLOCKED: refused, nothing to approve. A resolved card shows what happened.
import { h } from "../dom.js";
import { icon } from "../icons.js";

const META = {
  CAUTION: { icon: "warn", v: "warn", headline: "Needs your approval", desc: "The assistant wants to do this on your computer:", args: "Show exact arguments" },
  DANGEROUS: { icon: "danger", v: "danger", headline: "Can’t be undone — read it, then type yes", desc: "Runs on your computer with real effects:", args: "Show all characters" },
  BLOCKED: { icon: "blocked", v: "block", headline: "Refused by the safety policy", desc: "", args: "Show arguments" },
};

/** One line for the card: `edit path +3 −1` for file edits, the command for a shell call, else the server's description. */
export function describeCall(d) {
  const a = d.arguments || {};
  if (d.tool_name === "write_file" || d.tool_name === "replace_in_file") return d.description || `edit ${a.path || ""}`;
  if (d.tool_name === "run_command") return String(a.command || d.description || "").split("\n")[0].slice(0, 160);
  return d.description || d.tool_name;
}

/**
 * @param d {approval_id, tool_name, arguments, risk_level, description}
 * @param opts {status: "pending"|"approved"|"denied", note, onDecide(approve:boolean)}
 */
export function approvalCard(d, { status = "pending", note = "", onDecide = null } = {}) {
  const level = (d.risk_level || "CAUTION").toUpperCase();
  const m = META[level] || META.CAUTION;
  const danger = level === "DANGEROUS", blocked = level === "BLOCKED", resolved = status !== "pending";
  const args = d.arguments && Object.keys(d.arguments).length ? JSON.stringify(d.arguments, null, 2) : "";
  const full = d.tool_name === "run_command" && d.arguments ? String(d.arguments.command || "") : args;
  const card = h("div", { class: `l-approval ${danger && !resolved ? "danger" : ""} ${blocked ? "blocked" : ""} ${resolved ? "resolved" : ""}`.trim(), role: "group", "aria-label": `${level} approval` });

  const hd = h("div", { class: "hd" }, h("span", { class: `v-${m.v}`, style: { display: "inline-flex", background: "none" } }, icon(m.icon, danger ? 20 : 18)),
    h("span", { class: `l-risk v-${m.v}`, text: level }), h("strong", { text: blocked ? describeCall(d) : m.headline }));
  if (resolved && !blocked) {
    const ok = status === "approved";
    hd.append(h("span", { class: `res`, style: { color: ok ? "var(--ok)" : "var(--warn)" } }, icon(ok ? "check" : "x", 14), note || (ok ? "approved" : "denied · nothing ran")));
  }
  card.append(hd);
  if (!blocked) card.append(h("div", { style: { fontSize: "14px" } }, m.desc + " ", h("span", { class: "mono", text: describeCall(d) })));
  if (blocked) card.append(h("div", { style: { fontSize: "13px", color: "var(--color-text-2)" } }, d.description || "Refused by the safety policy. There is nothing to approve."));
  if (full) card.append(h("details", {}, h("summary", { text: m.args }), h("pre", { text: full })));

  if (!resolved && !blocked) {
    const approve = h("button", { class: "btn go", disabled: danger, onclick: () => onDecide && onDecide(true) }, icon("check", 16), danger ? "Approve and run" : "Approve");
    const deny = h("button", { class: "btn btn-secondary", onclick: () => onDecide && onDecide(false) }, "Deny");
    if (danger) {
      const input = h("input", { class: "input", id: "yes-" + d.approval_id, placeholder: "yes", autocomplete: "off", style: { fontFamily: "var(--font-mono)", maxWidth: "240px", minHeight: "44px" } });
      input.addEventListener("input", () => { approve.disabled = input.value !== "yes"; });         // the exact, case-sensitive word
      card.append(h("div", { class: "field" }, h("label", { for: input.id }, "Type ", h("strong", { class: "mono", text: "yes" }), " to enable Approve"), input));
    }
    card.append(h("div", { class: "btns" }, approve, deny));
  }
  return card;
}
