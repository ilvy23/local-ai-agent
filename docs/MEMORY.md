# How memory works

Everything on this page happens on your computer. Nothing is sent anywhere: facts, messages and vectors live in one
SQLite file (`data/agent.db` in the project folder, or under `NIGHTAGE_HOME`), and the models run in your own Ollama.

## What is stored

| What | Where | Used for |
|---|---|---|
| Your messages and the replies | `messages` | the chat history; also recalled by meaning |
| Facts about you ("has a cat called Mira") | `facts` (kind `personal`) | always shown to the model |
| Facts learned from web research | `facts` (kind `incidental`) | shown only when relevant |
| One vector per fact and message | `memory_items` + `memory_vectors` | finding things by meaning |
| What still has to be looked at | `distill_progress`, `learn_queue` | the queue below |

## When facts are saved

Facts are saved **while you chat**, not when you quit.

1. Every message you send is stored at once. That is the durable queue: for each session, `distill_progress` remembers
   the last message the fact extractor has seen, so "waiting" simply means "above that mark".
2. While the chat model is idle (you are reading or typing) a small background thread takes the waiting messages
   (at most 8 per call, `memory.distill_batch_messages`) and asks `models.background` for lasting facts about you. It
   never starts while the chat model is working on a turn, and it pauses while a game runs or the GPU is busy
   (`background.*` settings, the same governor as the other background work).
3. Facts and the new mark are written in **one transaction**. A crash before that loses nothing (the messages are
   still waiting and are redone); a crash after it saves nothing twice. Two Nightage processes on the same database
   cannot save the same batch twice either.
4. The quality rules are unchanged: only your own words are read (never tool output or the assistant's replies),
   path-like or tool-echo "facts" are rejected, duplicates and paraphrases of known facts are dropped, and an
   overlap check decides whether a new fact replaces an old one ("lives in Hamburg" -> "lives in Munich").
   Facts learned from the web are only kept when the sources were verified.

`memory.distill_every_turns` (default 1) sets how many of your messages must be waiting before the chat's own
session is looked at; older sessions are taken at once.

## Quitting

Quitting never waits for a model and never starts a model call. It stops the background thread (waiting at most
`memory.exit_wait_seconds`, default 3, for a step that is already running), queues tool results for later and prints
at most one dim line: `N messages will be remembered in the background`. The rest is finished

* at the **next start** of a chat (quietly, in the background; it prints nothing unless it fails), or
* with `nightage memory process` (for a scheduled job or the worker), or
* by the web server after `/clear` (web chats that never use `/clear` wait for one of the other two).

If Ollama is not running, the work simply stays queued and is retried later.

## How long it takes

Measured on a CPU-only laptop (7 GB RAM, Ollama 0.40, `qwen3.5:4b` for chat and background, about 5 tokens/s), a session
with three messages:

| | Before (fact distillation when you quit) | Now |
|---|---|---|
| Time spent waiting when quitting | 46 s behind a spinner | 0.3 s (the whole process, including unloading models) |
| When the facts are in the database | after those 46 s | about 20 s after the reply that followed the message, while the chat is still open |

Extraction costs the same model time as before (a quicker GPU shortens it proportionally), but it now runs while you read
and type, in small batches of only the new messages instead of one call over the whole session, and never at the same
moment as the chat model's own work.

## The memory (embedding) model

The memory model turns text into vectors so that "What is my cat called?" finds "has a cat called Mira" although they
share few words. It runs on the **CPU** (not the GPU), so it competes with the chat model for RAM, not VRAM. Without
it everything is still saved and found by keywords.

`nightage embed list` shows this table with what is installed; `nightage embed use <tag>` switches.

| Model (`models.embed`) | Download | RAM loaded | Vector | Who it is for |
|---|---|---|---|---|
| `all-minilm` | 46 MB | 47 MB | 384 | English only - smallest, fastest. 512-token input |
| `nomic-embed-text` | 274 MB | 371 MB | 768 | English only - small, good all-rounder. 2K input |
| `embeddinggemma:300m-qat-q4_0` | 239 MB | 295 MB | 768 | Multilingual (100+ languages, e.g. German, French, Spanish) - small. 2K input, needs Ollama 0.11.10+ |
| `granite-embedding:278m` | 563 MB | 546 MB | 768 | Multilingual - mid size. 512-token input |
| `qwen3-embedding:0.6b` | 639 MB | 1180 MB | 1024 | Multilingual (100+ languages) - very long inputs (32K) |
| `bge-m3` (default) | 1.2 GB | 1186 MB | 1024 | Multilingual (100+ languages, e.g. German, French) - best for non-English chats, the biggest |

Known but not recommended: `granite-embedding:30m` (English, 63 MB; separated unrelated sentences poorly in a quick
probe), `mxbai-embed-large` (English, 670 MB), `snowflake-arctic-embed2` (multilingual, 1.2 GB),
`paraphrase-multilingual` (563 MB, older). `nightage embed list --all` shows them.

How the numbers were obtained: download size and context window from the tag pages on ollama.com (checked
2026-10-27); vector size and RAM measured with Ollama 0.40 on a CPU-only machine (embed a few sentences, read
`/api/ps`; they vary a little by version); languages from the models' ollama.com pages, except `all-minilm` and
`nomic-embed-text`, which are English models according to their upstream model cards (the Ollama pages do not say).
Quality was **not** benchmarked: a four-pair probe (English, German, French, plus an unrelated pair) ranked every
related pair above the unrelated one for all six, which only shows they work, not which is best.

The installers pick for you from your RAM and language (English: `all-minilm` below 4 GB RAM, else `nomic-embed-text`;
other languages: `embeddinggemma` below 16 GB, else `bge-m3`) and show the same list; `--embed TAG` and `--no-embed`
skip the question. `nightage doctor` shows the label of the model in use, warns when it is large next to the free RAM
and names the recommended one.

### Switching the model

Vectors of different models cannot be compared - even two models with the same size - so `nightage embed use <tag>`
pulls the model if missing (after asking), probes its vector size, sets `models.embed` and re-embeds everything
(`nightage reembed`, which it calls). It refuses while a chat is open. If you edit `models.embed` by hand instead,
Nightage refuses to write mismatching vectors and tells you to run `nightage reembed <model>`; facts are still saved and
found by keyword until you do.

## Looking at it

```
nightage memory list        # every fact it holds
nightage memory search Q    # by meaning, falling back to keywords
nightage memory forget ID   # take one back
nightage memory prune       # drop junk facts (paths, tool output)
nightage memory repair      # embed facts that were saved while the memory model was unavailable
nightage memory process     # finish queued fact extraction now
nightage memory check       # save and recall a throwaway fact to prove it works
nightage doctor             # includes the memory.roundtrip check
```

`memory check` and `memory.roundtrip` use a temporary database that is deleted afterwards and say exactly why when
recall falls back to keywords ("embedding model X is not installed", "Ollama answered 500", a similarity cutoff that
is set too high, ...).

## Privacy

Facts are plain text in your own database file; delete them with `memory forget`, or delete the file. Nothing leaves
the computer except what you configure yourself (a remote Ollama in `ollama.host`, or the web search tool when you
use it).
