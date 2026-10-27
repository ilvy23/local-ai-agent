"""A slash command, a hook and a data folder working together.

  /devnote buy milk      appends a line to <data_dir>/notes.txt
  /devnotes           shows the last notes and how many turns finished this run
Settings (config.yaml):  plugins: {dev-notes: {show_last: 5}}
"""

from datetime import datetime


def register(api):
    notes = api.data_dir() / "notes.txt"          # a folder only this plugin writes to
    turns = {"done": 0}

    def add(args: str) -> str:
        text = args.strip()
        if not text:
            return "usage: /devnote <text>"
        with notes.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M}  {text}\n")
        return "saved"

    def show(args: str) -> str:
        lines = notes.read_text(encoding="utf-8").splitlines() if notes.exists() else []
        last = lines[-int(api.config("show_last", 5)):]
        head = f"{len(lines)} note(s), {turns['done']} turn(s) finished this run"
        return head + ("\n" + "\n".join(last) if last else "")

    def turn_end(data):                           # data is read-only; exceptions here are logged and ignored
        if data.get("outcome") == "done":
            turns["done"] += 1

    api.command("devnote", "Save a quick note", add, usage="/devnote <text>")
    api.command("devnotes", "Show your saved notes", show)
    api.on("turn_end", turn_end)
