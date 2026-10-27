# Nightage — web frontend

Hand-written ES modules, no build step, no CDN at runtime (fonts and libraries are self-hosted in `fonts/` and `vendor/`,
licences next to them). The design is: black first, violet accent, round shapes (tokens in `css/tokens.css`
= `lantern-tokens-v2.css`, dark default and `[data-theme="light"]`; component classes in `css/broadsheet.css`; application styles in `css/lantern.css`). Contract for the data: `docs/design/02-event-component-contract.md`.

```
index.html            loads css + js/main.js
css/                  fonts.css  broadsheet.css (component classes)  tokens.css (v2 tokens, dark/light, aliases)  lantern.css (app)  viewkit.css  s-<screen>.css (per screen)
js/main.js            boot, hash router (#/chat/12, #/memory, ...), loads js/screens/<name>.js on demand
js/shell.js           frame: sidebar, header (title, context meter, model + host pills, theme), tabs on phone, inspector
js/api.js             api.get/post/put/patch/del(path, body) -> JSON, ApiError(status, detail); stream(path, body, handlers, signal) for SSE
js/store.js           shared state: theme, density, connection, model, host, ctx; set({...}), subscribe(fn)
js/dom.js             h(tag, props, ...children), svg(), clear(), mono(), fmtDateTime(), relTime(), debounce()
js/icons.js           icon(name, size) -> <svg>  (names: see ICON_NAMES)
js/ui.js              toast(), dialog(), confirmDialog(), copyText(), download()
js/md.js              renderMarkdown(text, {onCite, claims, stray}) -> element (sanitised)
js/components/        trust.js verdict.js source.js sources.js block.js approval.js filecard.js mediaplayer.js composer.js
js/viewkit.js         generic data-driven screens (renderView / mountView), see "View kit" below
js/views-data/        static design data per area (sessions, whatsapp, code, apps, knowledge, research, tools, settings, ops, patterns) + _h.js builders
js/screens/           one module per destination
```

## A screen

```js
// js/screens/example.js
import { h } from "../dom.js";
import { api } from "../api.js";

export async function mount(root, ctx) {
  ctx.css("s-example");                       // optional: loads /css/s-example.css once
  ctx.shell.setTitle("Example");              // header title (default: the destination name)
  const timer = setInterval(refresh, 5000);
  ctx.onCleanup(() => clearInterval(timer));  // called when the user navigates away
  root.replaceChildren(h("div", { class: "l-page" }, h("h2", { text: "Example" })));
}
```

`ctx` = `{ name, params, shell, css(name), onCleanup(fn), navigate("#/x"), route }`. `shell` has `setTitle`, `openInspector(el)`,
`closeInspector()`, `isPhone()`, `loadSessions()`, `sessions`. A screen must handle its loading, empty and error states.

## Rules

* Values from the server go in via `textContent` / attributes (`h()` does this); never `innerHTML`.
* Colours, fonts, spacing, radius, shadows only from the tokens (`var(--color-*)`, `--ok/--warn/--danger/--info/--block/--neutral` + `-bg`,
  `--space-*`, `--font-*`). Semantic colours always come with an icon and a word.
* Layout works at 390 px (phone) and 1280 px: breakpoint `@media (max-width: 820px)`. Touch targets >= 44 px on phone (`--hit` is 36 px desktop).
* Destructive actions use `confirmDialog(..., { danger: true })`.
* Respect `prefers-reduced-motion` (already global in tokens.css).
* Class prefix `l-` for application classes; a screen's own CSS file uses `.s-<screen>-…` selectors so screens never collide.

## Tokens

`css/tokens.css` is the only place colours, radii, shadows and type sizes are defined. Dark is the default (`:root`, `[data-theme="dark"]`),
light is `[data-theme="light"]`; the store default is `dark`. Accent = violet (`--color-accent` fills with white text, `--color-accent-text` for text on black,
`--color-accent-soft`/`-soft-2` for selected states, `--color-focus` for the ring). Semantic pairs `--ok/--info/--warn/--danger/--block/--neutral` (+ `-bg`);
info is cyan, never purple. Charts use `--series-1..6` and must also differ by dash and marker (see `components/block.js`). Radii: `--radius-sm/md/lg/xl`
10/14/18/26, `--radius-control` 12, `--radius-pill`. Old names (`--color-process-yellow`, `--font-heading-weight`) are aliased at the end of the file.
`tests/test_web_static.py` fails if CSS/JS use a variable that is not defined.

## View kit (`js/viewkit.js`)

Screens made of settings rows, cards, tables, jobs, logs, wizards and so on are declared as data and rendered by `viewkit`. The full section
reference is the comment at the top of `js/viewkit.js`; the shapes are the ones in the design's `lantern-views.js` (`js/views-data/*.js` use the same
builder names: `G R B J T LD C LG W D E S CV F A TH BL BR X tab`, from `js/views-data/_h.js`, re-exported by `viewkit.js`).

```js
import { mountView, G, S } from "../viewkit.js";
import { api } from "../api.js";

export async function mount(root, ctx) {
  mountView(root, ctx, { title: "Operations", tabs: [{
    k: "worker", l: "Background worker", poll: 5000,                      // silent reload every 5 s (not while the user is typing)
    load: async () => { const w = await api.get("/api/worker"); return [   // sections are plain objects
      S([["State", w.state, ""], ["Queue", String(w.queued), "jobs"]]),
      G("Worker", [{ label: "Enabled", ctl: "sw", val: w.enabled, def: true, onChange: (v) => api.put("/api/worker", { enabled: v }) }]),
      { k: "job", title: "Embedding", pct: w.pct, st: "running", el: w.el, eta: w.eta, poll: async () => api.get("/api/worker/job"), acts: [{ l: "Stop", onClick: () => api.post("/api/worker/stop") }] },
    ]; },
  }] });
}
```

* `tab.s` static sections or `tab.load()` async (loading box, error box with retry, `tab.poll`, `tab.needs`, `tab.empty`). `view.notice` is a banner shown above every tab.
* Controls: `onChange(value, row)` may be async; throwing (or returning `{error}`) shows the message under the row, `{val}` replaces the value. "changed" and Reset follow `val` vs `def`; `onReset(row)` overrides the default reset. `f: "c r p e:msg"` flags and `restart`/`changed`/`error` fields are supported.
* Buttons: `"Label"`, `"+Primary"`, `"-Danger"` or `{ l, primary, danger, icon, onClick(ctx) }`; async handlers show busy, errors become a toast. A section-level `onAct(label, ctx)` serves string buttons. Without any handler a click toasts "Demo data, not connected yet".
* Any section may have `poll: async () => patch` + `pollMs` (job and log stop themselves when done); `ld` takes `detailFor(item, i)` for live detail panes; `block` and `approval` sections take real server data (`{ block }`, `{ call, onDecide }`).
* Layout is CSS only (`css/viewkit.css`, breakpoint 820 px, controls 44 px high). Call `ctx.css("viewkit")` (done by `mountView`).
* Placeholder screens (`sessions knowledge apps research tools ops`, plus the sampler at `#/patterns`) import `views-data/<area>.js` and show a "Preview" banner; a follow-up replaces sections with live data and removes the banner.
* `components/thinking.js` is the working indicator (status line in chat, `think` sections).
