// Static design data for the "settings" area (static sample data). A screen replaces sections with live data:
// see js/viewkit.js (renderView) and static/README.md.
import { G, R, tab } from "./_h.js";

export default { title: 'Settings', tabs: [
  tab('persona', 'Personality', [G('Personality', [R('Name', 'How the assistant refers to itself.', 'tx', 'Nightage', 'Nightage'), R('Style prompt', 'Added to every chat.', 'ta', 'Calm, direct. Answer first, then detail.', '(built-in)', 'c')])], ['Persona settings']),
  tab('models', 'Models', [G('Models', [R('Default model', '', 'sel:qwen3.8:27b-text|gemma3:12b', 'qwen3.8:27b-text', 'qwen3.8:27b-text'), R('Keep model loaded (min)', '', 'nm', '30', '10', 'c'), R('GPU mode', 'Chosen on Models & GPU.', 'tx', 'Both GPUs', '', 'r')])], ['Model/GPU settings']),
  tab('memory', 'Memory & chat', [G('Memory & chat', [R('Remember facts', '', 'sw', 'on', 'on'), R('Recall per answer', 'Facts shown in the trace.', 'nm', '5', '5'), R('Wrap-up when context is full', '', 'sw', 'on', 'on'), R('Show reasoning', 'The model’s thoughts are never shown.', 'sw', 'off', 'off', 'p')])]),
  tab('perf', 'Performance', [G('Performance', [R('Pause while gaming', '', 'sw', 'on', 'on'), R('GPU cap', '', 'sl', '80', '60', 'c'), R('RAM cap (GB)', '', 'nm', '96', '16', 'c e:Must be 62 or less, the installed RAM.')])], ['Resource limits']),
  tab('coding', 'Coding', [G('Coding', [R('Command timeout (s)', '', 'nm', '30', '30', 'r'), R('Max repair attempts', '', 'nm', '2', '2'), R('Auto-commit', '', 'sw', 'off', 'off')])]),
  tab('wa', 'WhatsApp', [G('WhatsApp', [R('Minimum messages for a profile', '', 'nm', '50', '50'), R('Blur when idle', 'Hides names and messages after a pause.', 'sw', 'off', 'off', 'p'), R('Sending', 'Blocked by policy.', 'sw', 'off', 'off')])]),
  tab('safety', 'Safety', [G('Safety', [R('Dangerous actions', '', 'seg:Always ask|Never run', 'Always ask', 'Always ask'), R('Allowed folders', 'Managed on Projects.', 'tx', '3 folders', ''), R('Block list', '', 'ta', '~/.ssh\n~/.gnupg', '')])]),
  tab('trust', 'Source trust', [G('Source trust', [R('Trust sensitivity', 'Strict rates more pages low or avoid.', 'seg:Lenient|Balanced|Strict', 'Balanced', 'Balanced'), R('Show passage in source cards', '', 'sw', 'on', 'on')])]),
] };
