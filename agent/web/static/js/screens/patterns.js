// Patterns: the component sampler (design data only, no server calls); open it at #/patterns.
import { mountView } from "../viewkit.js";
import area from "../views-data/patterns.js";

export async function mount(root, ctx) {
  mountView(root, ctx, { ...area, id: "patterns" });
}
