// Static design data for the "code" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { A, B, BL, BR, D, F, G, LD, LG, R, S, T, TH, tab } from "./_h.js";

export default { title: 'Code agent', tabs: [
  tab('goal', 'Start a goal', [
    G('Goal', [R('Folder', 'Where the agent may work.', 'pa', '~/code/rps', ''), R('Access level', 'What runs without asking in this folder.', 'seg:Ask|Medium|Auto|Greenlist', 'Auto', 'Ask', 'c'), R('Goal', 'Plain words.', 'ta', 'Add a rock-paper-scissors game with tests.', ''), R('Spec files', 'Attach files that describe the task.', 'pa', 'SPEC.md', '(none)')]),
    B('warn', 'Auto access: blast radius', 'Edits files and runs scoped commands inside ~/code/rps without asking. Anything outside always asks.'),
    G('Behaviour', [R('Effort', '', 'seg:Mid|Max', 'Mid', 'Mid'), R('Diagnose first', 'Look before changing.', 'sw', 'on', 'on'), R('Staged mode / autopilot', '', 'seg:Staged|Autopilot', 'Staged', 'Staged', 'c'), R('Auto-commit', 'Commit after each passing stage.', 'sw', 'off', 'off')]),
    G('Budget', [R('Max actions', '', 'nm', '200', '200'), R('Max seconds', '', 'nm', '3 600', '3 600'), R('Max tokens', '', 'nm', '500 000', '500 000')]),
    G('Models', [R('Executor model', '', 'sel:qwen3.8:27b-text|gemma3:12b', 'qwen3.8:27b-text', 'qwen3.8:27b-text'), R('Prose model', 'Second CPU model for graphic or mature text passages.', 'sel:none|gemma2:9b', 'none', 'none'), R('Escalation (tier-3) model', '', 'sel:none|llama3.3:70b-q4', 'llama3.3:70b-q4', 'none', 'c'), R('Escalation total budget', '', 'nm', '60 000', '60 000')]),
    G('Sandbox & safety', [R('Command timeout (s)', '', 'nm', '30', '30', 'r'), R('Memory limit (GB)', '', 'nm', '8', '8'), R('Max repair attempts', '', 'nm', '2', '2'), R('Protect patterns', 'Files the agent may never touch.', 'ta', '.env\n*.pem\nsecrets/**', '(none)', 'c')]),
    A([['Start', 'play', 1], ['Save as preset', 'download', 0]]),
  ], ['Goal start payload with all options; model list; protect patterns']),
  tab('run', 'Run view', [
    B('warn', 'Needs your decision', 'Add the “rich” package? It would be installed into the project environment.', ['+Allow', 'Skip']),
    A([['Pause', 'clock', 0], ['Stop', 'stop', 0], ['Resume', 'play', 0]]),
    TH('coding', 'rps/game.py'),
    S([['Actions', '148/200', ''], ['Time', '45/60 min', ''], ['Tokens', '312k/500k', ''], ['Checks', '2 failed', 'retry 1/2']]),
    T('Timeline', ['Time', 'Stage / step', 'Result'], [['m:00:06', 'Stage 1 · edit rps/game.py +126', 'ok:done'], ['m:00:14', 'Stage 1 · check pytest -q -x', 'danger:2 failed'], ['m:00:22', 'Verification gate', 'warn:retry 1/2'], ['m:00:28', 'Escalation: tier-3 model', 'info:running'], ['m:00:31', 'Context masked · 4 old outputs', 'neutral:compaction']], []),
    BL('diff'),
    BR('Why slow: per-step timing', [['Prefill', 38, '4 m 10 s'], ['Generation', 31, '3 m 24 s'], ['Tools', 22, '2 m 25 s'], ['Waiting for you', 9, '59 s']]),
  ], ['Live stage/step timeline, timing breakdown, budget meters, decision prompts']),
  tab('sessions', 'Sessions', [
    F([['Search', 'goal or folder', 'search'], ['Folder', 'all', 'folder']]),
    T('Runs', ['Goal', 'Folder', 'When', 'Result', ''], [['Add rock-paper-scissors game', 'm:~/code/rps', 'now', 'info:running', 'Resume'], ['Fix flaky timer test', 'm:~/code/nightage', '2 d', 'ok:passed', 'Export'], ['Port settings loader', 'm:~/code/lantern', '7 d', 'warn:stopped', 'Resume']], ['Export trajectory']),
    G('Housekeeping', [R('Prune sessions older than', 'days', 'nm', '30', '30')]),
  ], ['Cross-folder session list, resume, prune, trajectory export']),
  tab('policy', 'Policy editor', [
    T('code-policy rules', ['Rule', 'Pattern', 'Effect'], [['1', 'm:git status', 'ok:allow'], ['2', 'm:pytest *', 'ok:allow'], ['3', 'm:rm -r *', 'danger:ask · DANGEROUS'], ['4', 'm:curl * | sh', 'block:blocked'], ['5', 'm:~/.ssh/**', 'block:blocked']], ['Add rule']),
    { k: 'test', title: 'Test a command', cmd: 'rm -r ~/Documents/old', tier: 'danger:DANGEROUS', why: 'Matches rule 3: deletes files, cannot be undone.' },
  ], ['Rule CRUD and a risk-tier test endpoint']),
  tab('eval', 'Evaluation', [
    T('code-eval runs', ['Task set', 'Pass rate', 'When'], [['basic-py (20)', 'ok:18/20 · 90 %', '2 d'], ['refactor (12)', 'warn:8/12 · 67 %', '5 d']], ['Run eval']),
    T('Model comparison (code-model)', ['Model', 'Pass rate', 'Median time'], [['qwen3.8:27b-text', 'ok:90 %', '6 m'], ['gemma3:12b', 'warn:71 %', '4 m']], []),
    LG('Per-task log: refactor/07', ['I: stage 1 started', 'W: test_move failed (1/3)', 'E: gave up after 2 repairs']),
  ], ['Eval run list, per-task logs, model comparison']),
  tab('checkpoints', 'Checkpoints', [
    LD('Git checkpoints', [['c1a9e2f · tests green', '14:22', 'ok:latest'], ['9be04d1 · game loop', '14:05', ''], ['4a7f0c8 · scaffold', '13:50', '']], 'c1a9e2f', [['Files', '3'], ['Diff vs previous', '+120 −4'], ['Diff vs now', 'none']], ['+Restore', 'Diff between…']),
  ], ['Checkpoint list, restore, diff between two']),
  tab('memory', 'Project memory', [
    G('~/code/rps', [R('Notes', 'Kept per repository.', 'ta', 'Tests live in rps/. Use uv.', '(empty)'), R('Check that passes', 'Command that proves a change is good.', 'tx', 'pytest -q -x', '(none)', 'c')]),
    D('Forget this project', 'Removes notes and the check command for ~/code/rps.', 'forget', 'Forget project'),
  ], ['Per-repository notes + check command']),
  tab('stats', 'Stats', [
    S([['Runs', '41', 'last 30 d'], ['Pass rate', '88 %', ''], ['Prefix-cache hit', '71 %', ''], ['Speed', '36 tok/s', 'median']]),
    BR('Speed trials', [['27b both GPUs', 100, '36.1 tok/s'], ['27b one GPU', 62, '22.4 tok/s'], ['12b one GPU', 197, '71 tok/s']]),
    T('Server benchmark (read-only)', ['Test', 'Result'], [['Cold prompt, 8k', 'm:14 s'], ['Warm prompt, 8k', 'm:1.8 s']], []),
  ], ['Run statistics, speed trials, prefix-cache report, benchmark results']),
] };
