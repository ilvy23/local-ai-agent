# Security

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Use GitHub's private reporting instead:
on the repository page open **Security -> Report a vulnerability**. Include what you found, how to
reproduce it and the version (`nightage version`). You will get an answer as soon as a maintainer can
look at it; this is a small project, so please be patient. Once a fix is released the report may be
made public with credit to you, if you want it.

## What this program can do on your machine

Nightage is an AI agent that runs locally. By design it can:

- run shell commands and read, write, move and delete files (the coding agent works inside the folder
  you point it at, in a throw-away git worktree where possible);
- make network requests: web search and page fetches when you ask for them (`/web`, research), and calls to
  your Ollama server. Pages are fetched through a guard that refuses local, private and VPN addresses;
- start long-running background work (research, code runs) and, if you enable it, small web apps.

Because a language model decides what to run, **treat it like a very fast, sometimes confidently wrong
colleague with a shell**. Do not point it at folders or machines you cannot afford to damage, and do not
run it as root.

## Safety mechanisms (guard rails, not a sandbox)

- Every tool call is risk-classified: safe things run, risky things ask for approval, dangerous things
  are refused outright. Folder access is `ask` by default; `auto`/`medium`/greenlist grants are explicit and
  per folder, and your home directory and system folders cannot be granted.
- Every command and tool call is written to a local audit log (`nightage audit`).
- Edits can be undone (`/undo`), and inside git repos a checkpoint ref is made before the first edit.
- The coding agent cannot edit its own tests to fake a pass and re-checks its work before saying "done".
- Secrets are redacted from the text sent back to the model and from exported trajectories.

These reduce mistakes; they are not a security boundary against a hostile model or hostile web content.
Web pages and tool output are untrusted input and can contain prompt injection. Review approval prompts.

## Where your data lives (local only)

- Chats, facts, the vector index and the audit log are in one SQLite file under `data/` in the project
  folder. Configuration is `config.yaml` and the folder `~/.config/agent`. Nothing is uploaded anywhere;
  the only outgoing traffic is what you ask for (web search/fetch) and the Ollama address you configured.
- If you point `ollama.host` at another machine, prompts go to that machine: use a network you trust
  (LAN, VPN) and keep Ollama off the public internet.
- The optional web GUI requires an access token (`nightage web-token`) on every API call. Do not expose it
  to the internet. Keep the token file private (it is created readable by you only).
- Do not commit `config.yaml`, `data/` or your token; they are in `.gitignore`.

## Supported versions

Only the latest release / main branch receives fixes.
