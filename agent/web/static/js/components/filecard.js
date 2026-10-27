// Cards for files the assistant sends (file_ready events / stored file_card rows): video (with subtitles), audio (with
// now-playing tags), image, download link. Ported from the previous GUI, including its iOS Safari workarounds.
import { h } from "../dom.js";
import { icon } from "../icons.js";
import { api } from "../api.js";
import { albumCard, canPlay, downloadLink } from "./mediaplayer.js";

const fileFormat = (name) => { const e = (name || "").split(".").pop(); return e ? e.toUpperCase() : "FILE"; };
const MIME = { mp4: "video/mp4", m4v: "video/mp4", webm: "video/webm", mkv: "video/x-matroska", mov: "video/quicktime", mp3: "audio/mpeg", flac: "audio/flac",
  ogg: "audio/ogg", opus: "audio/ogg", wav: "audio/wav", m4a: "audio/mp4", aac: "audio/aac" };
const guessMime = (name, tag) => MIME[(name || "").split(".").pop().toLowerCase()] || "";
const meta = (d) => h("div", { class: "meta" }, h("span", { class: "n", text: d.filename || "file" }), h("span", { class: "mono", text: [fileFormat(d.filename), d.size].filter(Boolean).join(" · ") }));

export function fileCard(d) {
  if (d.kind === "album") return albumCard(d);
  const card = h("div", { class: "l-file", dataset: { kind: d.kind || "file" } });
  // single video/audio: if the browser can't play the type (or errors on it) offer the download instead of a dead player
  const guard = (el, tag) => {
    const item = { url: d.url, filename: d.filename };
    if (!canPlay(tag, d.mime || guessMime(d.filename, tag))) { el.replaceWith(downloadLink(item, "this browser can't play this format")); return; }
    el.addEventListener("error", () => { if (el.getAttribute("src") && !el.nextSibling?.classList?.contains("l-mp-fallback")) el.after(downloadLink(item, "couldn't play this file")); });
  };
  if (d.kind === "video") {
    const video = h("video", { controls: true, playsinline: true, "webkit-playsinline": "true", preload: "metadata" });
    if (d.subtitle_url) {                       // the <track> must exist BEFORE src is set, and be forced to "showing" (iOS Safari)
      const track = h("track", { kind: "subtitles", src: d.subtitle_url, label: d.subtitle_label || "Subtitles", default: true });
      video.append(track);
      try { track.track.mode = "showing"; track.addEventListener("load", () => { track.track.mode = "showing"; }); } catch (_) { /* no TextTrack API */ }
    }
    video.src = d.url;
    card.append(video, meta(d)); guard(video, "video");
  } else if (d.kind === "image") {
    card.append(h("img", { src: d.url, alt: d.filename || "image", loading: "lazy" }), meta(d));
  } else if (d.kind === "audio") {
    if (d.title) {
      card.append(h("div", { style: { display: "flex", gap: "12px", alignItems: "center" } },
        d.art_url ? h("img", { src: d.art_url, alt: "", style: { width: "64px", height: "64px", objectFit: "cover", flex: "none" } })
          : h("span", { class: "l-ib", style: { width: "64px", height: "64px", background: "var(--color-raised)" } }, icon("music", 28)),
        h("div", { style: { minWidth: 0 } }, h("div", { class: "mono", text: "Now playing" }), h("strong", { text: d.title }),
          d.artist ? h("div", { style: { fontSize: "13px", color: "var(--color-text-2)" }, text: d.artist }) : null,
          d.album ? h("div", { style: { fontSize: "12px", color: "var(--color-text-3)" }, text: d.album }) : null)));
    }
    const audio = h("audio", { controls: true, src: d.url, preload: "metadata" });
    card.append(audio, meta(d)); guard(audio, "audio");
    if ("mediaSession" in navigator) audio.addEventListener("play", () => { try { navigator.mediaSession.metadata = new MediaMetadata({ title: d.title || d.filename || "", artist: d.artist || "", album: d.album || "", artwork: d.art_url ? [{ src: d.art_url }] : [] }); } catch (_) { /* no MediaMetadata */ } });
  } else {
    card.append(h("div", { class: "meta" }, icon("download", 18), h("a", { class: "n", href: d.url, target: d.kind === "link" ? "_blank" : null, rel: "noopener", download: d.kind === "download" ? (d.filename || "") : null, style: { color: "var(--color-accent-text)" }, text: d.filename || "download" }),
      h("span", { class: "mono", text: d.size || "" })));
  }
  return card;
}

/** Image generation runs for minutes in the background: show progress and poll until the image is ready. */
export function imageJobCard(data, onReady) {
  const status = h("span", { class: "mono", text: "queued" });
  const bar = h("div", { class: "l-progress" }, h("i", { style: { width: "4%" } }));
  const card = h("div", { class: "l-card" },
    h("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, icon("image", 16), h("strong", { style: { fontSize: "14px", flex: 1 }, text: "Generating image" }), status),
    bar, h("span", { style: { fontSize: "13px", color: "var(--color-text-2)" }, text: `About ${Math.round((data.eta_s / 60) * 10) / 10} min on this hardware. Replies are slower until it finishes.` }));
  const started = Date.now(), eta = Math.max(1, data.eta_s) * 1000;
  let timer = null;
  const tick = async () => {
    bar.firstChild.style.width = Math.min(95, Math.round(((Date.now() - started) / eta) * 100)) + "%";
    try {
      const j = await api.get(`/api/images/${data.job_id}`);
      status.textContent = j.status;
      if (j.status === "done" && j.url) { card.replaceWith(fileCard({ kind: "image", url: j.url, filename: j.filename, size: j.size })); onReady && onReady(); return; }
      if (j.status === "failed") { status.textContent = "failed"; card.append(h("p", { class: "l-note v-danger", style: { background: "none" }, text: j.error || "Image generation failed." })); return; }
    } catch (e) {
      if (e.status === 404) { status.textContent = "lost"; return; }                // the server restarted and forgot the job
    }
    if (Date.now() - started < eta * 4) timer = setTimeout(tick, 4000);
  };
  timer = setTimeout(tick, 2500);
  card.cleanup = () => clearTimeout(timer);
  return card;
}
