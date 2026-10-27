// Static design data for the "research" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { B, C, F, G, LD, R, T, tab } from "./_h.js";

export default { title: 'Research, trust & web', tabs: [
  tab('rules', 'Source trust', [
    F([['Search domains', 'rust-lang.org', 'search']]),
    T('Your domain rules', ['Domain', 'Rule', 'Effect'], [['m:doc.rust-lang.org', 'ok:allow', 'High trust'], ['m:best-dev-tools-2026.example', 'block:deny', 'Hidden from answers'], ['m:lwn.net', 'info:tier T2', 'Medium']], ['Add domain', 'Import', 'Export']),
    LD('Why this rating', [['blog.rust-lang.org', 'high', 'ok:high']], 'blog.rust-lang.org', [['Reasons', 'official project blog · has author · has date'], ['Your rule', 'none'], ['Tier', 'T1']], []),
  ], ['Domain rule CRUD, import/export, explainer']),
  tab('feedback', 'Feedback history', [
    T('Per-source feedback', ['Domain', 'You said', 'When'], [['m:dev-digest.example', 'warn:poor', '18 Oct'], ['m:blog.rust-lang.org', 'ok:good', '18 Oct']], []),
  ], ['Feedback history list']),
  tab('web', 'Web search', [
    T('Engine chain', ['Engine', 'Status', ''], [['SearXNG (http://127.0.0.1:8888)', 'ok:reachable', 'Test'], ['DuckDuckGo', 'ok:reachable', 'Test'], ['Brave (key set)', 'warn:rate limited', 'Set key · Test']], []),
    C('Connectors', [{ t: 'Wikipedia', s: 'enabled', sw: 1 }, { t: 'PyPI', s: 'enabled', sw: 1 }, { t: 'npm', s: 'enabled', sw: 1 }, { t: 'arXiv', s: 'off', sw: 0 }, { t: 'GitHub', s: 'enabled', sw: 1 }, { t: 'Hacker News', s: 'off', sw: 0 }, { t: 'Weather', s: 'enabled', sw: 1 }, { t: 'Exchange rates', s: 'enabled', sw: 1 }]),
    G('Cache', [R('Page cache TTL (days)', '', 'nm', '7', '7'), R('Cache size', 'Read-only: 184 MB', 'tx', '184 MB', '')]),
    B('ok', 'SSRF guard on', 'Local and private addresses are never fetched. Read-only.'),
  ], ['Engine status/test, connector toggles/tests, cache size']),
  tab('verify', 'Verify history', [
    T('“Verify this answer” runs', ['Answer', 'Before', 'After'], [['Rust 1.90 release date', 'warn:Single source', 'info:Likely'], ['GPU price drop', 'danger:Conflicting', 'danger:Conflicting']], []),
  ], ['Verify run history']),
] };
