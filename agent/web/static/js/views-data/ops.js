// Static design data for the "ops" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { A, B, BR, F, G, J, LG, R, S, T, tab } from "./_h.js";

export default { title: 'Operations', tabs: [
  tab('governor', 'Resource governor', [S([['GPU', '71 %', 'cap 80'], ['RAM', '28/62 GB', 'cap 96'], ['CPU', '23 %', '']]), G('Thresholds', [R('Pause background work above (GPU %)', '', 'nm', '85', '85', 'p'), R('Resume below (GPU %)', '', 'nm', '60', '60', 'p')], true)], ['Governor state + thresholds']),
  tab('worker', 'Background worker', [T('Queue', ['Job', 'State'], [['Memory distiller', 'ok:running'], ['Skill curator', 'neutral:queued'], ['Re-embed', 'warn:paused: gaming']], ['Pause all']), J('Memory distiller', 'Reading 12 chats', 55, 'running', '1:44', '1 min left', [])], ['Worker queue + pause']),
  tab('panel', 'Panel & status', [S([['Chat server', 'up', 'ok'], ['Model server', 'up', 'ok'], ['Worker', 'running', ''], ['Uptime', '3 d 4 h', '']]), BR('Model placement (read-only)', [['GPU 0', 90, '21.4 / 24 GB'], ['GPU 1', 82, '19.6 / 24 GB']])], ['Read-only dashboards']),
  tab('logs', 'Logs', [F([['Level', 'all', 'list'], ['Search', '', 'search']]), LG('Server log', ['I: 14:03:02 web_search “rust 1.90 release date”', 'I: 14:03:08 read_page blog.rust-lang.org', 'W: 14:03:19 timeout users.rust-lang.example', 'E: 14:31:44 pytest exit 1'])], ['Log tail stream']),
  tab('backup', 'Backup', [J('Backup to ~/Backups', 'Database and memory', 100, 'done', '0:42', '', [], ['Open folder']), G('Backup', [R('Schedule', '', 'seg:Off|Daily|Weekly', 'Daily', 'Off', 'c'), R('Keep last (n)', '', 'nm', '7', '7')]), A([['Back up now', 'download', 1], ['Restore…', 'undo', 0]])], ['Backup job + restore']),
  tab('sync', 'DB sync', [B('info', 'Read-only', 'Shows whether two machines hold the same data.'), T('Machines', ['Machine', 'Last seen', 'State'], [['desktop', 'now', 'ok:in sync'], ['laptop', '2 d ago', 'warn:behind by 41 changes']], [])], ['Sync status between machines']),
] };
