// Static design data for the "knowledge" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { A, F, G, J, R, S, T, tab } from "./_h.js";

export default { title: 'Knowledge', tabs: [
  tab('search', 'Memory search', [
    F([['Semantic search', 'what did I say about GPUs', 'search']]),
    T('Results', ['Fact', 'Similarity', 'Kind'], [['Runs Debian on a desktop PC', 'm:0.74', 'neutral:personal'], ['Main machine is “desktop”', 'm:0.41', 'neutral:personal'], ['Building a rock-paper-scissors game', 'm:0.22', 'neutral:incidental']], []),
  ], ['Semantic search with scores']),
  tab('maintain', 'Maintenance', [
    S([['Embedding model', 'bge-m3', 'local'], ['Index', 'healthy', 'ok'], ['Vectors', '3 412', '']]),
    J('Re-embed memory', 'Rebuilds the search index', 38, 'running', '2:10', '4 min left', []),
    A([['Repair index', 'sliders', 0], ['Export memory', 'download', 0], ['Import memory…', 'download', 0]]),
  ], ['Index health, repair, import/export']),
  tab('curator', 'Skill curator', [
    T('Curator report', ['Skill', 'Finding'], [['Convert flac to opus', 'neutral:archived (5 months unused)'], ['Rotate VPN key', 'warn:stale'], ['Fix pytest import', 'info:duplicate of “uv pytest”']], ['+Run curator now']),
    G('Schedule', [R('Run the curator', '', 'seg:Off|Weekly|Daily', 'Weekly', 'Weekly', 'c')]),
  ], ['Curator report, run now, schedule']),
  tab('notes', 'Notes', [
    G('Notes', [R('Working notes', 'Scratch space the assistant can read.', 'ta', '', '(empty)'), R('Session notes', 'Per chat — see Sessions.', 'ta', '(per chat)', ''), R('Project notes', 'Per folder — see Code agent.', 'ta', '(per folder)', '')]),
  ], ['Working-notes storage']),
] };
