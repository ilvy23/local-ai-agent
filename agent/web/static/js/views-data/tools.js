// Static design data for the "tools" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { B, C, F, LD, T, tab } from "./_h.js";

export default { title: 'Tools & audit', tabs: [
  tab('tools', 'Tools', [
    T('Every tool the agent can call', ['Tool', 'Description', 'Risk', 'Enabled', 'Last use'], [['m:web_search', 'Search the web', 'ok:SAFE', 'ok:on', '14:03'], ['m:read_page', 'Read a page', 'ok:SAFE', 'ok:on', '14:04'], ['m:edit_file', 'Change a file in an allowed folder', 'warn:CAUTION', 'ok:on', '14:12'], ['m:run_command', 'Run a shell command', 'danger:DANGEROUS', 'ok:on', '14:31'], ['m:write_file', 'Write outside a folder', 'block:BLOCKED', 'block:off', '—']], []),
  ], ['Tool registry with risk tier, enabled flag, last use']),
  tab('audit', 'Audit log', [
    F([['Search', 'tool, argument', 'search'], ['Risk', 'any', 'shield-half'], ['Decision', 'any', 'check']]),
    T('Actions', ['Time', 'Tool', 'Arguments', 'Risk', 'Decision', 'Result'], [['m:14:31', 'm:run_command', 'm:pytest -q -x', 'ok:SAFE', 'ok:auto', '56 passed'], ['m:14:15', 'm:run_command', 'm:rm -r ~/Documents/old', 'danger:DANGEROUS', 'warn:denied', 'not run'], ['m:14:16', 'm:write_file', 'm:~/.ssh/config', 'block:BLOCKED', 'block:refused', 'nothing ran']], ['Export CSV']),
    LD('Detail', [['14:15 · run_command', 'denied', 'warn:denied']], 'run_command · rm -r ~/Documents/old', [['Risk', 'DANGEROUS'], ['Decision', 'denied by you, 14:15'], ['Result', 'not run']], []),
  ], ['Audit search/filter/detail, export']),
  tab('packs', 'Tool packs', [
    C('Optional packs', [{ t: 'Diagnostics', s: 'Read system state', sw: 0, p: 1 }, { t: 'Network audit', s: 'Needs authorisation: scope + confirmation', sw: 0, p: 1 }], true),
    B('warn', 'PROPOSAL: authorisation step', 'A pack that needs a scope asks for it, with a typed word, before it can be switched on. Exact fields are open.'),
  ], ['Pack list, enable, scope/authorisation fields']),
] };
