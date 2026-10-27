// Static design data for the "apps" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { A, C, G, J, LD, R, S, T, tab } from "./_h.js";

export default { title: 'Apps', tabs: [
  tab('images', 'Images', [
    C('Generated', [{ t: 'a lantern in rain', s: 'seed 4821 · 1024² · 38 s', img: 1 }, { t: 'lantern on a desk', s: 'seed 77 · 768² · 21 s', img: 1 }, { t: 'salt road at night', s: 'seed 9 · 1024² · 40 s', img: 1 }]),
    J('Queue: “a lantern in snow”', 'Step 14 of 30', 47, 'running', '0:20', '0:23 left', [], ['Cancel']),
    G('Settings', [R('Default size', '', 'seg:768|1024', '1024', '1024', 'p'), R('Steps', '', 'nm', '30', '30', 'p')], true),
  ], ['Gallery with prompt/seed/size/time; queue; settings']),
  tab('files', 'Shared files', [
    T('Files the assistant shared', ['File', 'Type', 'Expires', ''], [['m:night-drive.mp3', 'audio', 'in 6 d', 'Open · Download'], ['m:report.pdf', 'file', 'in 2 d', 'Open · Download'], ['m:old-render.png', 'image', 'warn:expired', '—']], []),
  ], ['Shared-media list with expiry']),
  tab('prompt', 'Prompt studio', [
    LD('Saved prompt chats', [['Nix flake review', 'today', ''], ['Cover letter', 'Tue', '']], 'Nix flake review', [['Strength', 'Medium'], ['Refine rounds', '2']], ['+Resume', '-Delete']),
    G('Refine loop', [R('Your feedback', 'Say what to change; it revises.', 'ta', 'shorter, keep the output format', '')]),
    T('Max anonymise: trace', ['Original', 'Replaced with'], [['Anna Meier', 'm:[PERSON_1]'], ['anna@example.com', 'm:[EMAIL_1]']], []),
  ], ['Saved prompt chats, refine loop, anonymise trace']),
] };
