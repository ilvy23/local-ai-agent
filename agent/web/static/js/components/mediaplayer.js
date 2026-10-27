// Album cards (file_ready with kind "album"): audio playlist, photo gallery with lightbox, video list. Native <audio>/<video>
// do the playing (Range streaming comes from /api/files/<token>); this file adds the list, next/prev, Media Session and
// the lightbox. A type the browser cannot play shows a download link instead of a dead player.
import { h } from "../dom.js";
import { icon } from "../icons.js";

const fmtTime = (s) => (s > 0 && isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}` : "");
const fmtSize = (n) => { if (!n) return ""; const u = ["B", "KB", "MB", "GB"]; let i = 0; while (n >= 1024 && i < 3) { n /= 1024; i++; } return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; };

/** Can this browser play `mime` as `tag` ("audio" | "video")? canPlayType is "" for a definite no. */
export const canPlay = (tag, mime) => !mime || document.createElement(tag).canPlayType(mime) !== "";

/** Download fallback row, used when a track/video cannot be played. */
export const downloadLink = (item, note) => h("div", { class: "l-mp-fallback" }, icon("download", 16),
  h("a", { href: item.url, download: item.filename || "" , text: item.filename || item.title || "download" }), note ? h("span", { class: "mono", text: note }) : null);

function mediaSession(item, d, handlers) {
  if (!("mediaSession" in navigator)) return;
  try {
    const art = d.cover_url ? [{ src: d.cover_url }] : [];
    navigator.mediaSession.metadata = new MediaMetadata({ title: item.title || "", artist: item.artist || "", album: d.name || "", artwork: art });
    for (const [action, fn] of Object.entries(handlers)) { try { navigator.mediaSession.setActionHandler(action, fn); } catch (_) { /* unsupported action */ } }
  } catch (_) { /* MediaMetadata missing */ }
}

function iconBtn(name, label, onClick, extra = {}) {
  return h("button", { type: "button", class: "l-ib l-mp-btn", "aria-label": label, title: label, onclick: onClick, ...extra }, icon(name, 20));
}

/** A playlist driven by one media element (audio or video). `list` rows are buttons: Enter/Space plays, arrows move focus. */
function playlist(d, tag) {
  const items = d.items || [];
  const el = h(tag, { controls: true, preload: "metadata", ...(tag === "video" ? { playsinline: true } : {}) });
  let cur = -1, order = items.map((_, i) => i), shuffle = false;
  const rows = items.map((it, i) => h("button", { type: "button", class: "l-mp-row", onclick: () => play(i), dataset: { i } },
    h("span", { class: "no mono", text: String(i + 1) }), h("span", { class: "t", text: it.title || it.filename }),
    it.artist ? h("span", { class: "a", text: it.artist }) : null,
    h("span", { class: "mono d", text: fmtTime(it.duration) || fmtSize(it.size) })));
  const now = h("div", { class: "l-mp-now" });
  const fallback = h("div", { class: "l-mp-fb" });
  const step = (dir) => { if (!items.length) return; const pos = order.indexOf(cur); play(order[(pos + dir + order.length) % order.length]); };
  function play(i) {
    const it = items[i]; if (!it) return;
    cur = i; fallback.replaceChildren();
    rows.forEach((r, k) => { r.classList.toggle("on", k === i); r.setAttribute("aria-current", k === i ? "true" : "false"); });
    now.textContent = it.title || it.filename;
    if (!canPlay(tag, it.mime)) { el.removeAttribute("src"); el.load(); fallback.append(downloadLink(it, "this browser can't play this format")); return; }
    el.src = it.url; el.play().catch(() => { /* autoplay blocked: the controls are right there */ });
    mediaSession(it, d, { play: () => el.play(), pause: () => el.pause(), previoustrack: () => step(-1), nexttrack: () => step(1),
      seekto: (e) => { el.currentTime = e.seekTime; } });
  }
  el.addEventListener("ended", () => { const pos = order.indexOf(cur); if (pos < order.length - 1) play(order[pos + 1]); });
  el.addEventListener("error", () => { if (cur >= 0 && el.getAttribute("src")) { fallback.replaceChildren(downloadLink(items[cur], "couldn't play this file")); } });
  const shuf = iconBtn("shuffle", "Shuffle", () => {
    shuffle = !shuffle; shuf.setAttribute("aria-pressed", String(shuffle));
    order = items.map((_, i) => i);
    if (shuffle) { for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; } }
  }, { "aria-pressed": "false" });
  const list = h("div", { class: "l-mp-list", role: "group", "aria-label": "Tracks", onkeydown: (e) => {
    const k = e.key, rs = rows, at = rs.indexOf(document.activeElement);
    if (k === "ArrowDown" && at < rs.length - 1) { rs[at + 1].focus(); e.preventDefault(); }
    else if (k === "ArrowUp" && at > 0) { rs[at - 1].focus(); e.preventDefault(); }
    else if (k === "n" || k === "N") step(1);
    else if (k === "p" || k === "P") step(-1);
  } }, ...rows);
  const bar = h("div", { class: "l-mp-bar" }, iconBtn("prev", "Previous track", () => step(-1)), iconBtn("next", "Next track", () => step(1)), shuf, now);
  return { el, bar, list, fallback, play };
}

function audioAlbum(d) {
  const p = playlist(d, "audio");
  const cover = d.cover_url ? h("img", { class: "l-mp-cover", src: d.cover_url, alt: "" }) : h("span", { class: "l-mp-cover ph" }, icon("music", 28));
  return h("div", { class: "l-mp-audio" },
    h("div", { class: "l-mp-head" }, cover, h("div", { class: "txt" }, h("div", { class: "mono", text: "Album" }), h("strong", { text: d.name }), h("div", { class: "mono", text: `${d.count || d.items.length} tracks · ${d.size || ""}` }))),
    p.el, p.bar, p.fallback, p.list);
}

function videoAlbum(d) {
  const p = playlist(d, "video");
  return h("div", { class: "l-mp-video" }, h("strong", { text: d.name }), p.el, p.bar, p.fallback, p.list);
}

function photoAlbum(d) {
  const items = d.items || [];
  const open = (i) => lightbox(items, i);
  return h("div", { class: "l-mp-photos" },
    h("div", { class: "l-mp-head" }, h("div", { class: "txt" }, h("strong", { text: d.name }), h("div", { class: "mono", text: `${d.count || items.length} photos · ${d.size || ""}` }))),
    h("div", { class: "l-mp-grid" }, ...items.map((it, i) => h("button", { type: "button", class: "l-mp-thumb", "aria-label": `Open ${it.title || it.filename}`, onclick: () => open(i) },
      h("img", { src: it.url, alt: it.title || "", loading: "lazy", decoding: "async" })))));
}

/** Full-screen viewer: arrows / swipe / buttons move, Escape closes, focus returns to the thumbnail. */
function lightbox(items, start) {
  const back = document.activeElement;
  let i = start, x0 = null;
  const img = h("img", { alt: "" });
  const cap = h("div", { class: "mono cap" });
  const show = (n) => { i = (n + items.length) % items.length; img.src = items[i].url; img.alt = items[i].title || ""; cap.textContent = `${i + 1} / ${items.length}  ${items[i].title || ""}`; };
  const close = () => { document.removeEventListener("keydown", onKey, true); box.remove(); back && back.focus && back.focus(); };
  const onKey = (e) => {
    if (e.key === "Escape") { close(); e.stopPropagation(); }
    else if (e.key === "ArrowRight") show(i + 1);
    else if (e.key === "ArrowLeft") show(i - 1);
  };
  const box = h("div", { class: "l-mp-lightbox", role: "dialog", "aria-modal": "true", "aria-label": "Photo viewer",
    onpointerdown: (e) => { x0 = e.clientX; },
    onpointerup: (e) => { if (x0 !== null && Math.abs(e.clientX - x0) > 50) show(i + (e.clientX < x0 ? 1 : -1)); x0 = null; } },
    h("div", { class: "stage" }, img), cap,
    iconBtn("x", "Close", close, { class: "l-ib l-mp-btn close" }),
    iconBtn("chevr", "Previous photo", () => show(i - 1), { class: "l-ib l-mp-btn prev" }),
    iconBtn("chevr", "Next photo", () => show(i + 1), { class: "l-ib l-mp-btn next" }));
  document.addEventListener("keydown", onKey, true);
  document.body.append(box); show(i);
  box.querySelector(".close").focus();
}

export function albumCard(d) {
  const kind = d.album_kind || "audio";
  const body = kind === "image" ? photoAlbum(d) : kind === "video" ? videoAlbum(d) : audioAlbum(d);
  return h("div", { class: "l-file l-mp", dataset: { kind: "album", album: kind } }, body);
}
