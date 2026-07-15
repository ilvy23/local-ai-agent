# companion

A fully-local AI companion CLI. Runs entirely against a local
[Ollama](https://ollama.com) instance — no data leaves your machine.

It remembers past conversations, can use your PC (shell, files, system
stats) behind a safety gate, and searches the live web when a question
needs current information.

## Prerequisites

- [Ollama](https://ollama.com) running locally: `ollama serve`
- Models pulled in Ollama (defaults, editable in `config.yaml`):
  - `dolphin3:8b` — chat + background tasks (fact distillation)
  - `bge-m3` — embeddings (multilingual, 1024-dim)

The embedding dimension is detected from the model and stored in the DB on
first use, so you can swap the embed model and re-index with `companion
reembed <model>`.

## Install

```
uv sync
```

## Usage

```
companion chat              # start a new chat session
companion resume [id]       # resume the most recent (or a given) session
companion sessions          # list past sessions
companion menu              # interactive menu over all commands
companion memory list       # list remembered facts
companion memory search Q   # search facts (semantic, falls back to substring)
companion memory add TEXT   # manually add a fact
companion memory forget ID  # forget a fact by id
companion memory prune      # drop junk facts (paths, tool output, timestamps)
companion audit             # view the tool/command audit log
companion panel             # live status panel (machine, Ollama, data)
```

### Web search

End any chat message with `/web` (or `/search`) to force a live web search
for it — the agent shows each site as it visits it and answers with cited
sources:

```
you> what changed in the latest python release /web
```

The agent can also search on its own when a question needs current info.

### Using your PC

The agent can run shell commands and read/write files through a tool layer
with a risk classifier: safe commands run automatically, riskier ones ask
for confirmation, and dangerous ones are blocked. Everything it runs is
recorded in the audit log (`companion audit`).

## Data

Everything is stored locally in `data/companion.db` (SQLite, WAL mode).
Nothing is synced or sent anywhere except to your local Ollama instance —
and to the web, only when you explicitly trigger a search.

## License

MIT — see [LICENSE](LICENSE).
