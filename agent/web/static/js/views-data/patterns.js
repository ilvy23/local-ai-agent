// Static design data for the "patterns" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { B, D, E, G, J, LD, LG, R, W, tab } from "./_h.js";

export default { title: 'Patterns', tabs: [
  tab('kit', 'Controls', [{ k: 'ui' }]),
  tab('patterns', 'Patterns', [
    G('Settings group', [R('Setting label', 'Help text explains the effect in plain words.', 'sw', 'on', 'off', 'c'), R('Number with error', 'Validation sits under the label.', 'nm', '96', '16', 'c e:Must be 62 or less.'), R('Needs restart', '', 'sel:a|b', 'a', 'a', 'r'), R('Proposal row', 'Unsure whether it exists.', 'tx', 'value', '', 'p')]),
    J('Long-running job card', 'Progress, elapsed, ETA, stop, log tail', 58, 'running', '2:10', '1:35 left', ['step 58/100 · batch ok']),
    LD('List + detail', [['First item', 'subtitle', 'ok:ready'], ['Second item', 'subtitle', '']], 'First item', [['Key', 'Value']], ['+Primary', 'Secondary']),
    W(['One', 'Two', 'Three'], 1, [R('Field', '', 'seg:A|B', 'A', '')], 'Summary of choices before you confirm.'),
    LG('Terminal-style log view', ['I: info line', 'W: warning line', 'E: error line']),
    D('Danger zone', 'Type the word to enable the button.', 'delete', 'Delete'),
    E('Empty state with CLI equivalent', 'Nothing here yet. The terminal command does the same.', 'nightage memory repair'),
    B('warn', 'Permission banner', 'Edits files in ~/code/rps without asking. Everything else asks.'),
  ]),
] };
