// Static design data for the "whatsapp" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { A, B, C, CV, F, G, J, LD, R, S, T, X, tab } from "./_h.js";

export default { title: 'WhatsApp', tabs: [
  tab('chats', 'Chats', [
    B('info', 'Read-only', 'Nothing is ever sent from here. Sending is blocked by policy.'),
    F([['Keyword', 'search messages', 'search'], ['Contact', 'any', 'user'], ['Date range', 'last 30 days', 'clock'], ['Blur when idle', 'Off', 'eyeoff']]),
    LD('Chats', [['Alex', 'See you at 7? · 14:02', 'neutral:contact'], ['Family', 'Mum: Photos are up · 12:40', 'neutral:group'], ['Team chat', 'Release is tagged · Mon', 'neutral:group'], ['Plumber', 'Thursday works · Sun', 'neutral:contact']],
      'Alex', [['14:02 Alex', 'See you at 7?'], ['13:58 you', 'Great. I’ll bring the cable.'], ['13:55 Alex', 'Are we still on for tonight?'], ['13:40 Alex', '[image · placeholder, tap to load]']], ['Older messages', 'Open in search']),
  ], ['Chat list, paged messages, search by keyword/contact/date']),
  tab('sync', 'Sync & ingest', [
    S([['Last sync', '14:02', 'today'], ['Messages', '48 211', 'in the index'], ['Errors', '2', 'last 24 h'], ['Bridge', 'running', 'ok']]),
    J('Embedding messages for semantic search', '48 211 of 61 004', 79, 'running', '12:08', '3 min left', ['batch 94/120 · 400 messages']),
    A([['Start bridge', 'play', 0], ['Stop bridge', 'stop', 0], ['Re-ingest all', 'refresh', 0], ['Import from backup file…', 'download', 0]]),
    T('Recent errors', ['Time', 'Error'], [['m:13:58', 'warn:Media not downloaded (expired)'], ['m:09:12', 'danger:Bridge lost connection, retried once']], []),
  ], ['Bridge/ingest status, start/stop, re-ingest, import backup, embedding progress']),
  tab('profiles', 'Profiles & analysis', [
    B('warn', 'Runs a big model on this PC', 'Deep analysis uses the large model and takes several minutes. Nothing leaves the PC.'),
    LD('People', [['Alex', '1 204 messages', 'ok:ready'], ['Family group', '8 311 messages', 'ok:ready'], ['Plumber', '14 messages', 'warn:too few']],
      'Alex', [['Summary', 'Friend. Plans evenings and trips together.'], ['Tone', 'Warm, brief, jokes'], ['Topics', 'Cycling · work · weekends'], ['Timeline', '2023 met · 2024 weekly · 2026 daily'], ['Red flags', 'None found'], ['Predictions', 'Likely to reply within 1 h (p)']], ['+Run deep analysis…', 'Refresh profile']),
    J('Deep analysis: Alex', 'Reading 1 204 messages', 34, 'running', '4:10', '8 min left', [], ['Stop']),
    G('Thresholds', [R('Minimum messages for a profile', 'Below this the person is skipped.', 'nm', '50', '50', 'c')]),
  ], ['Per-contact profile, red flags, predictions, deep-analysis job']),
  tab('digest', 'Digest & nudges', [
    C('Today', [{ t: 'Daily digest', s: '14 chats · 3 need a reply', tag: 'ok:ready' }, { t: 'Daily note', s: 'Evening plan with Alex; plumber Thursday', tag: 'ok:ready' }, { t: 'Moments', s: '2 notable: trip dates agreed, new job news', tag: 'info:2 new' }]),
    LD('Nudges', [['Reply to Alex about 7 pm', 'asked 2 h ago', 'warn:due'], ['Follow up with the plumber', 'Thursday', 'neutral:soon']], 'Reply to Alex', [['Why', 'Asked a question, no reply yet'], ['Due', 'today 16:00']], ['+Accept', 'Dismiss', 'Snooze 1 h']),
    G('Schedule & delivery', [R('Generate digest at', '', 'tx', '07:30', '07:30'), R('Deliver as', 'Where you see it.', 'seg:In app|Notification', 'In app', 'In app'), R('Nudges', 'Reminders to reply or follow up.', 'sw', 'on', 'on')]),
  ], ['Digest, daily note, moments, nudges (accept/dismiss/snooze), schedule']),
  tab('persona', 'Persona & auto-reply', [
    B('danger', 'Nothing is ever sent', 'Auto-reply is shown for review only. Sending stays blocked unless enabled outside this app.'),
    C('Personas', [{ t: 'Alex persona', s: 'LoRA · last build 12 Oct', tag: 'ok:ready' }, { t: 'Family persona', s: 'not trained', tag: 'neutral:empty' }]),
    S([['Train set', '3 912', 'messages'], ['Validation', '435', 'messages'], ['Last build', '12 Oct', '2026']]),
    J('Training: Alex persona (LoRA)', 'Step 640 of 1 000', 64, 'running', '36:20', '20 min left', [], ['Stop']),
    CV('Loss', [2.8, 2.4, 2.1, 1.9, 1.75, 1.62, 1.55, 1.5, 1.46, 1.43], 'checkpoints at 200 · 400 · 600'),
    T('Checkpoints & evaluation', ['Checkpoint', 'Test score', 'State'], [['m:step 600', '0.71', 'ok:best'], ['m:step 400', '0.66', 'neutral:kept'], ['m:step 200', '0.52', 'neutral:kept']], []),
    { k: 'chat', title: 'Try the persona (mini chat)', msgs: [['you', 'Are you free tonight?'], ['persona', 'Yeah, free after 7. Same place?']] },
    T('Auto-reply per contact', ['Contact', 'State'], [['Alex', 'block:off · sending blocked'], ['Family', 'block:off · sending blocked']], []),
  ], ['Persona list, dataset stats, LoRA train job + loss series, evaluation results, persona chat, per-contact state']),
  tab('safety', 'Safety', [
    B('danger', 'Sending is blocked by policy', 'This banner stays on every WhatsApp screen.'),
    G('Policy (read-only)', [R('Send messages', 'Blocked by the product.', 'sw', 'off', 'off', 'c')]),
    { k: 'approval', level: 'DANGEROUS' },
    X('If sending is ever enabled elsewhere', 'The approval card shows the exact message and recipient and needs you to type “yes”. This app never offers a send button.'),
  ], ['Policy state endpoint']),
] };
