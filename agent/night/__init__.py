"""`nightage night`: an unattended coding run for the night (private edition only).

You hand it a queue of prompts and a time budget (default 8 hours). It does NOT stop because the model is stuck:
  * prompts are revealed ONE AT A TIME (queue.py); the next one only appears when the current one is done or its time slice is spent,
  * the first minutes settle a CHARTER of commands (charter.py) that you confirm once; after that it runs without asking,
  * every attempt is a separate `nightage code --autopilot` process (runner.py) that a watchdog can kill (timeout / no progress),
  * after an attempt the project's own tests decide (governor.py), good states become git checkpoints (gitbox.py),
  * a failed attempt escalates: resume -> fresh context + handoff note -> roll back to the last green checkpoint -> small steps,
  * the run ends with a consistent state: the last green checkpoint, the red work kept under a tag, and NIGHT_REPORT.md.
Nothing here is imported by the public edition: the package is on the export deny-list and the CLI is registered optionally."""
