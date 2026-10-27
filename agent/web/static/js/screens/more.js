// "More" (design S-More): the destinations that do not fit in the phone's four tabs, plus the theme and density switches.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { NAV } from "../shell.js";
import { state, set, effectiveTheme } from "../store.js";

const SKIP = new Set(["chat", "board", "memory"]);

function seg(label, name, options, current, onPick) {
  return h("div", { class: "s-more-row", role: "radiogroup", "aria-label": label },
    h("span", { text: label }),
    h("div", { class: "seg" }, options.map(([value, text, ic]) =>
      h("label", { class: "seg-opt s-more-opt" },
        h("input", { type: "radio", name, value, checked: value === current, onchange: () => onPick(value) }), ic ? icon(ic, 15) : null, text))));
}

export async function mount(root, ctx) {
  ctx.css("s-more");
  ctx.shell.setTitle("More");
  root.replaceChildren(h("div", { class: "s-more" },
    h("nav", { class: "s-more-list", "aria-label": "More destinations" }, NAV.filter(([k]) => !SKIP.has(k)).map(([k, label, ic]) =>
      h("a", { class: "s-more-item", href: "#/" + k }, icon(ic, 22), h("span", { text: label }), icon("chevr", 16)))),
    h("div", { class: "s-more-prefs" },
      seg("Theme", "s-more-theme", [["light", "Light", "sun"], ["dark", "Dark", "moon"]], effectiveTheme(), (v) => set({ theme: v })),
      seg("Density", "s-more-density", [["comfortable", "Comfortable"], ["compact", "Compact"]], state.density, (v) => set({ density: v })))));
}
