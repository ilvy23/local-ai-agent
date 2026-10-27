// Lazy loaders for the self-hosted libraries in /vendor (UMD builds that set a global). Mermaid is 3.5 MB: only
// fetched when the first diagram appears.
const loading = new Map();

function script(src) {
  if (!loading.has(src)) {
    loading.set(src, new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src; s.async = true;
      s.onload = () => resolve(); s.onerror = () => { loading.delete(src); reject(new Error("could not load " + src)); };
      document.head.append(s);
    }));
  }
  return loading.get(src);
}

export const ensureMarked = () => (window.marked ? Promise.resolve(window.marked) : script("/vendor/marked.umd.js").then(() => window.marked));
export const ensureHljs = () => (window.hljs ? Promise.resolve(window.hljs) : script("/vendor/highlight.min.js").then(() => window.hljs));

let mermaidReady = null;
export function ensureMermaid(dark) {
  mermaidReady = mermaidReady || script("/vendor/mermaid.min.js").then(() => window.mermaid);
  return mermaidReady.then((m) => {
    const cs = getComputedStyle(document.documentElement), v = (n) => cs.getPropertyValue(n).trim();
    // "base" theme driven by the Lantern tokens, so diagrams sit on black (or light) like the rest of the UI
    m.initialize({ startOnLoad: false, securityLevel: "strict", theme: "base", fontFamily: "Source Serif 4, Georgia, serif", themeVariables: {
      darkMode: !!dark, background: v("--color-surface"), primaryColor: v("--color-raised-2"), primaryTextColor: v("--color-text"), primaryBorderColor: v("--color-accent-text"),
      secondaryColor: v("--color-raised"), tertiaryColor: v("--color-surface"), lineColor: v("--color-text-3"), textColor: v("--color-text"), mainBkg: v("--color-raised-2"),
      nodeBorder: v("--color-accent-text"), clusterBkg: v("--color-raised"), clusterBorder: v("--color-divider"), edgeLabelBackground: v("--color-surface"), noteBkgColor: v("--warn-bg"), noteTextColor: v("--warn") } });
    return m;
  });
}
