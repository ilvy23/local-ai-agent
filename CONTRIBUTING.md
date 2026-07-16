# Contributing

Thanks for your interest! This is a small, fully-local project and contributions
of any size are welcome — bug reports, docs, or code.

## Getting set up

```bash
git clone https://github.com/ilvy23/local-ai-agent
cd local-ai-agent
uv sync                 # installs deps + a matching Python
uv run agent menu       # try it
```

You'll also need [Ollama](https://ollama.com) running with the default models
(`ollama pull dolphin3:8b bge-m3`) to use the agent — but **not** to run the
tests.

## Running the tests

```bash
uv run pytest           # the full suite; no Ollama required
```

Tests marked `integration` hit a real local Ollama and are skipped by default:

```bash
uv run pytest -m integration    # opt in, needs Ollama
```

Please add a test with any behaviour change. Keep them small and dependency-free
(the suite uses plain `pytest`, no fixtures framework beyond the built-ins).

## Style

- Small, focused changes and PRs.
- Match the surrounding code; the project favours the standard library and a
  minimal dependency set.
- Write a short comment when a non-obvious shortcut is intentional.

## Ideas and feedback

Ideas are very welcome, and they don't have to be fully formed. Start a
[Discussion](https://github.com/ilvy23/local-ai-agent/discussions) — that's the
right place for "wouldn't it be cool if", questions, or telling me what annoyed
you. No template, no ceremony.

## Reporting bugs

Open an [issue](https://github.com/ilvy23/local-ai-agent/issues) with what you
ran, what you expected, and what happened (include your OS, Python version, and
the model you're using). Good first issues are labelled
[`good first issue`](https://github.com/ilvy23/local-ai-agent/labels/good%20first%20issue).

## Scope

The project is intentionally local-first: no telemetry, no cloud dependencies,
and the only network access is a web search the user explicitly triggers. Please
keep contributions within that spirit.
