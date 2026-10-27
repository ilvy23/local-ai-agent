// Static design data for the "sessions" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { B, C, D, F, G, J, LD, R, T, tab } from "./_h.js";

export default { title: 'Sessions & chat tools', tabs: [
  tab('manage', 'Sessions', [
    F([['Search sessions', 'Ctrl K', 'search'], ['Sort', 'Recent', 'list'], ['Pinned first', 'On', 'pin']]),
    LD('All sessions', [['Rust 1.90 release date', '24 messages · today 14:03', 'info:pinned'], ['Clean up old build folder', '9 messages · 2 h ago', ''], ['GPU context sizes', '31 messages · yesterday', ''], ['Debian package notes', '12 messages · Mon', 'neutral:forked']],
      'Rust 1.90 release date', [['Model', 'qwen3.8:27b-text'], ['Project', '~/code/rps'], ['Started', '18 Oct 2026 13:41'], ['Messages', '24 (web ×3, tools ×11)'], ['Notes', '2 notes · plan not set']],
      ['+Resume', 'Rename', 'Pin', 'Export Markdown', 'Export JSON', 'Fork from here', '-Delete…']),
    D('Delete all sessions', 'Removes every chat and message. Memory facts stay. This cannot be undone.', 'delete', 'Delete all sessions'),
  ], ['Session list with pin flag and export endpoints (Markdown/JSON)', 'Fork-from-message endpoint', 'Rename endpoint']),
  tab('turn', 'Per-turn controls', [
    T('Controls around one message', ['Control', 'Where', 'State'], [['Force web search', 'Composer “/web” toggle', 'ok:available'], ['Regenerate', 'Under the last answer', 'ok:available'], ['Edit last message', '↑ in an empty composer', 'ok:available'], ['Stop', 'Send button becomes Stop while running', 'ok:available'], ['Model for one message', 'Model chip in composer', 'ok:available'], ['Think harder / quick answer', 'Composer option, only if the model supports it', 'neutral:PROPOSAL']], [], false),
    B('info', 'Edit and regenerate create a branch', 'The previous answer is kept as an earlier version you can step back to. (Display only; behaviour as in the product today.)'),
  ], ['Model capability flag for “think harder”']),
  tab('slash', 'Slash commands', [
    T('Every command, with its screen', ['Command', 'What it does', 'Screen'], [['m:/model', 'Switch model', 'Model picker sheet'], ['m:/clear', 'Wrap up and start fresh', 'Confirm dialog'], ['m:/notes', 'Session notes', 'Notes tab'], ['m:/plan', 'Show the plan', 'Plan tab'], ['m:/project', 'Set the project folder', 'Folder picker'], ['m:/undo', 'Undo file changes', 'Undo list'], ['m:/auto-<dir>', 'Auto access for a folder', 'Folder access sheet'], ['m:/medium-<dir>', 'Medium access', 'Folder access sheet'], ['m:/greenlist-<dir>', 'Always-allow list', 'Folder access sheet'], ['m:/access', 'Show the access list', 'Folder access']], []),
    B('warn', 'Access commands change what runs without asking', 'The sheet shows the blast radius in plain words before it applies.'),
  ], ['List of commands with descriptions from the server']),
  tab('undo', 'Undo', [
    LD('File-change snapshots', [['c1a9e2f · before editing game.py', '3 files · 14:12', 'ok:current'], ['9be04d1 · before test run', '1 file · 14:05', ''], ['4a7f0c8 · session start', '0 files · 13:41', '']],
      'c1a9e2f', [['rps/game.py', '+126 −0'], ['rps/test_game.py', '+48 −0'], ['README.md', '+3 −4']], ['+Restore this snapshot', 'Diff vs now']),
    B('info', 'Restoring is itself undoable', 'A new snapshot is taken first.'),
  ], ['Snapshot list + restore + diff endpoints']),
  tab('notes', 'Notes & plan', [
    G('Session notes', [R('Notes for this chat', 'Kept with the session; the assistant can read them.', 'ta', 'Pin nixpkgs to the 24.11 channel. Ask about the second GPU later.', '(empty)', 'c')]),
    T('Plan checklist', ['Item', 'Status'], [['Pin the input in flake.nix', 'ok:confirmed'], ['Evaluate both machines', 'neutral:open'], ['Commit the lock file', 'neutral:open']], ['Edit plan']),
    G('Project notes', [R('Notes for ~/code/rps', 'Shared by every chat in this project.', 'ta', 'Check that passes: pytest -q -x. Tests live in rps/.', '(empty)', 'c')]),
  ], ['Session notes read/write; plan edit; project notes']),
  tab('files', 'Attachments & media', [
    T('Attached to this message', ['File', 'Size', 'Type', ''], [['m:flake.nix', '3.1 kB', 'text', 'Remove'], ['m:logs.zip', '1.2 MB', 'archive', 'Remove']], []),
    C('Sent back by the assistant', [{ t: 'night-drive.mp3', s: 'audio · 4.2 MB · expires in 6 d', img: 1 }, { t: 'lantern-rain.png', s: 'image · 1.1 MB · expires in 6 d', img: 1 }, { t: 'demo.mp4', s: 'video · 38 MB · expires in 2 d', img: 1 }]),
    J('Image generation: “a lantern in rain”', 'Queue position 1 of 3', 62, 'running', '1:12', '1:40 left', [], ['Cancel']),
    J('Image generation: “lantern on a desk”', 'Queued', 0, 'queued', '—', '—', [], ['Cancel']),
  ], ['Image job queue with ETA + cancel']),
] };
