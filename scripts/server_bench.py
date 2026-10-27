#!/usr/bin/env python3
"""A/B benchmark of server-side Ollama knobs (draft_num_predict, num_batch). stdlib + httpx only.

    python scripts/server_bench.py --dry-run                      # print plan + sample requests, sends nothing
    python scripts/server_bench.py --yes                          # run the default grid (draft 4,6,8 x batch 512,1024,2048)
    python scripts/server_bench.py --yes --draft 4,8 --batch 512,2048 --repeats 3 --json out.json

Needs the GPUs to be free: it sends real generation requests. Without --yes nothing is sent at all.
The baseline config (draft 4 / batch 512) is always run first and everything is compared to it.

Workloads (all fixed and deterministic: temperature 0, fixed seed, fixed prompts):
  W1 prefill       ~12k-token synthetic source text, num_predict 8. A warm call first, then the first ~40 tokens
                   are changed so the whole prompt is reprocessed. tok/s = prompt_eval_count / prompt_eval_duration.
  W2 decode-code   write a ~400 token Python function from a fixed spec. tok/s = eval_count / eval_duration.
  W3 decode-json   tool-call-like JSON (~200 tokens), same measurement.
Each configuration gets one warm-up request (model reload after a num_batch change) that is not counted.
Quality guard: the W2/W3 text of every config is compared (difflib) with the baseline text; speculative decoding
and batch size must not change output at temperature 0, so a ratio below --min-similarity is flagged.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import statistics
import sys
from typing import Any

import httpx

BASELINE = (4, 512)
SEED = 1234
WORKLOADS = ("prefill", "code", "json")
LABELS = {"prefill": "W1 prefill", "code": "W2 code", "json": "W3 json"}
CHARS_PER_TOKEN = 3.3          # rough, for sizing the synthetic prompt only; the real count comes from the server

CODE_SPEC = (
    "Write one complete Python function `merge_intervals(intervals, gap=0)` that takes a list of (start, end) "
    "integer tuples, sorts them, merges overlapping or touching intervals (touching means the gap between them "
    "is <= gap), validates input (raise ValueError when start > end), and returns the merged list. Include a "
    "docstring with two doctest examples, type hints, and inline comments. Output only the code, no prose."
)
JSON_SPEC = (
    "Output ONLY a JSON object, no prose, describing three tool calls to inspect a repository. Format: "
    '{"tool_calls": [{"name": "...", "arguments": {...}}, ...]}. Use the tools read_file(path, start, end), '
    "search(pattern, path, max_results) and run_tests(target, verbose). Give each call realistic, fully "
    "filled arguments and add a short \"reason\" string to every call."
)


# ---------------------------------------------------------------------------------------------------- helpers

def parse_int_list(text: str, name: str) -> list[int]:
    try:
        values = [int(p) for p in text.replace(" ", "").split(",") if p]
    except ValueError:
        raise SystemExit(f"bad --{name} value {text!r} (comma separated integers, e.g. 4,6,8)") from None
    if not values or any(v <= 0 for v in values):
        raise SystemExit(f"bad --{name} value {text!r} (positive integers only)")
    return list(dict.fromkeys(values))


def parse_float_list(text: str | None) -> list[float]:
    if not text:
        return []
    try:
        return [float(p) for p in text.replace(" ", "").split(",") if p]
    except ValueError:
        raise SystemExit(f"bad --temperature-list value {text!r}") from None


def build_grid(drafts: list[int], batches: list[int]) -> list[tuple[int, int]]:
    """All (draft, batch) pairs; the baseline pair is always first (added when the grid lacks it)."""
    grid = [(d, b) for d in drafts for b in batches]
    if BASELINE in grid:
        grid.remove(BASELINE)
    return [BASELINE, *grid]


def synthetic_source(tokens: int, seed: int = SEED, variant: int = 0) -> str:
    """Deterministic source-code-like text of about `tokens` tokens. `variant` changes only the first lines."""
    rng = random.Random(seed)
    names = ["load", "merge", "parse", "emit", "flush", "route", "scan", "bind", "probe", "split", "fold", "walk"]
    lines: list[str] = []
    target = int(tokens * CHARS_PER_TOKEN)
    size = 0
    n = 0
    while size < target:
        a, b, c = rng.choice(names), rng.choice(names), rng.randint(2, 97)
        block = (
            f"def {a}_{b}_{n}(items, limit={c}):\n"
            f"    \"\"\"{a.capitalize()} the {b} items up to {c} entries.\"\"\"\n"
            f"    out = []\n"
            f"    for i, item in enumerate(items):\n"
            f"        if i >= limit:\n"
            f"            break\n"
            f"        out.append(({a!r}, item, i * {rng.randint(3, 41)}))\n"
            f"    return out\n\n"
        )
        lines.append(block)
        size += len(block)
        n += 1
    header = (
        f"# build-{variant:04d} module-{rng.randint(0, 9) if variant else 0} reviewed by {'ab' * 4} on branch "
        f"{'x' * (1 + variant % 7)}\n"
    )
    return header + "".join(lines)


def prefill_messages(tokens: int, variant: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": synthetic_source(tokens, variant=variant) + "\nReply with the single word OK."}]


def workload_messages(kind: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": CODE_SPEC if kind == "code" else JSON_SPEC}]


def build_payload(model: str, messages: list[dict], draft: int, batch: int, num_ctx: int,
                  num_predict: int, temperature: float = 0.0) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "draft_num_predict": draft,
            "num_batch": batch,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "seed": SEED,
        },
    }


def median_of(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def tps(count: int | float, duration_ns: int | float) -> float | None:
    """tokens per second from a count and a duration in ns (None when the server gave no usable timing)."""
    if not count or not duration_ns or duration_ns <= 0:
        return None
    return float(count) / (float(duration_ns) / 1e9)


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def speedup(value: float | None, base: float | None) -> float | None:
    if value is None or not base:
        return None
    return value / base


# ---------------------------------------------------------------------------------------------------- HTTP

def stream_chat(client: httpx.Client, payload: dict[str, Any]) -> dict[str, Any]:
    """POST /api/chat (stream) and return {text, final}: the concatenated content and the last (done) chunk."""
    text: list[str] = []
    final: dict[str, Any] = {}
    with client.stream("POST", "/api/chat", json=payload) as response:
        if response.status_code >= 400:
            response.read()
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(f"server error: {chunk['error']}")
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                text.append(piece)
            if chunk.get("done"):
                final = chunk
                break
    return {"text": "".join(text), "final": final}


def draft_stats(final: dict[str, Any]) -> dict[str, Any]:
    """Whatever speculative-decoding counters the server put in the final chunk (key names vary by version)."""
    return {k: v for k, v in final.items() if any(w in k.lower() for w in ("draft", "accept", "specul"))}


def measure(client: httpx.Client, kind: str, args: argparse.Namespace, draft: int, batch: int, rep: int) -> dict[str, Any]:
    if kind == "prefill":
        payload = build_payload(args.model, prefill_messages(args.prefill_tokens, variant=rep + 1), draft, batch,
                                args.num_ctx, 8)
    else:
        payload = build_payload(args.model, workload_messages(kind), draft, batch, args.num_ctx,
                                400 if kind == "code" else 200)
    res = stream_chat(client, payload)
    final = res["final"]
    if kind == "prefill":
        rate = tps(final.get("prompt_eval_count", 0), final.get("prompt_eval_duration", 0))
    else:
        rate = tps(final.get("eval_count", 0), final.get("eval_duration", 0))
    return {"tps": rate, "text": res["text"], "draft": draft_stats(final),
            "prompt_tokens": final.get("prompt_eval_count"), "eval_tokens": final.get("eval_count")}


# ---------------------------------------------------------------------------------------------------- run

def run_config(client: httpx.Client, args: argparse.Namespace, draft: int, batch: int, log=print) -> dict[str, Any]:
    out: dict[str, Any] = {"draft": draft, "batch": batch, "workloads": {}}
    # warm-up: loads the model with this num_batch / num_ctx; not counted
    stream_chat(client, build_payload(args.model, workload_messages("json"), draft, batch, args.num_ctx, 8))
    for kind in WORKLOADS:
        if kind == "prefill":   # warm the prefix once; each measured call below then has a different first line
            stream_chat(client, build_payload(args.model, prefill_messages(args.prefill_tokens, 0), draft, batch,
                                              args.num_ctx, 8))
        runs = [measure(client, kind, args, draft, batch, r) for r in range(args.repeats)]
        rates = [r["tps"] for r in runs if r["tps"] is not None]
        out["workloads"][kind] = {
            "median": median_of(rates), "min": min(rates) if rates else None, "max": max(rates) if rates else None,
            "rates": rates, "text": runs[0]["text"] if kind != "prefill" else "",
            "draft_stats": runs[-1]["draft"], "prompt_tokens": runs[-1]["prompt_tokens"],
            "eval_tokens": runs[-1]["eval_tokens"],
        }
        log(f"  {LABELS[kind]:<11} median {_fmt(out['workloads'][kind]['median'])} tok/s  (n={len(rates)})")
    return out


def determinism_check(client: httpx.Client, args: argparse.Namespace, temps: list[float]) -> list[dict[str, Any]]:
    """Baseline config only: W2 twice per temperature, compare. Temperature 0 must be identical."""
    d, b = BASELINE
    rows = []
    for t in temps:
        pl = build_payload(args.model, workload_messages("code"), d, b, args.num_ctx, 400, temperature=t)
        first, second = stream_chat(client, pl)["text"], stream_chat(client, pl)["text"]
        rows.append({"temperature": t, "similarity": similarity(first, second)})
    return rows


def analyse(results: list[dict[str, Any]], min_similarity: float) -> dict[str, Any]:
    """Add speedup and similarity to every result and pick the recommendation. results[0] is the baseline."""
    base = results[0]
    for res in results:
        res["speedup"], res["similarity"] = {}, {}
        for kind in WORKLOADS:
            res["speedup"][kind] = speedup(res["workloads"][kind]["median"], base["workloads"][kind]["median"])
        for kind in ("code", "json"):
            res["similarity"][kind] = similarity(base["workloads"][kind]["text"], res["workloads"][kind]["text"])
        res["flagged"] = any(v < min_similarity for v in res["similarity"].values())
        sp = [v for v in res["speedup"].values() if v is not None]
        res["score"] = sum(sp) / len(sp) if sp else None
    ok = [r for r in results if not r["flagged"] and r["score"] is not None]
    best = max(ok, key=lambda r: r["score"]) if ok else None
    return {"best": best}


def _fmt(v: float | None, spec: str = ".1f") -> str:
    return "-" if v is None else format(v, spec)


def render_table(results: list[dict[str, Any]]) -> str:
    head = f"{'draft/batch':<12}" + "".join(f"{LABELS[k]:>18}" for k in WORKLOADS) + f"{'sim W2':>8}{'sim W3':>8}  note"
    lines = [head, "-" * len(head)]
    for r in results:
        cells = ""
        for k in WORKLOADS:
            w = r["workloads"][k]
            cells += f"{_fmt(w['median']) + ' x' + _fmt(r['speedup'][k], '.2f'):>18}"
        base = "  (baseline)" if (r["draft"], r["batch"]) == BASELINE else ""
        flag = "  DIFFERS from baseline" if r["flagged"] else ""
        lines.append(f"{str(r['draft']) + '/' + str(r['batch']):<12}{cells}"
                     f"{r['similarity']['code']:>8.3f}{r['similarity']['json']:>8.3f}{base}{flag}")
    lines.append("cells: median tok/s x speedup vs baseline 4/512")
    return "\n".join(lines)


def recommendation(best: dict[str, Any] | None, min_similarity: float) -> str:
    if best is None:
        return f"Recommendation: none (no config with similarity >= {min_similarity} and valid timings)"
    return (f"Recommendation: draft_num_predict={best['draft']} num_batch={best['batch']} "
            f"(mean speedup x{best['score']:.2f} vs 4/512, similarity >= {min_similarity})")


# ---------------------------------------------------------------------------------------------------- CLI

def default_model() -> str:
    try:
        from agent.coding import models
        name = models.current()
        if name:
            return name
    except Exception:
        pass
    return "qwen3.8:27b-text"


def default_url() -> str:
    try:
        from agent.llm import vulkan_base_url
        return vulkan_base_url()
    except Exception:
        return "http://127.0.0.1:11436"


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A/B benchmark of Ollama draft_num_predict / num_batch.")
    p.add_argument("--model", default=None, help="model tag (default: the agent's code model, else qwen3.8:27b-text)")
    p.add_argument("--url", default=None, help="Ollama base URL (default: the Vulkan server, AGENT_VULKAN_OLLAMA_URL)")
    p.add_argument("--draft", default="4,6,8", help="draft_num_predict values (default 4,6,8)")
    p.add_argument("--batch", default="512,1024,2048", help="num_batch values (default 512,1024,2048)")
    p.add_argument("--temperature-list", default=None, help="determinism sanity check at the baseline, e.g. 0,0.25")
    p.add_argument("--repeats", type=int, default=3, help="repeats per workload (default 3)")
    p.add_argument("--prefill-tokens", type=int, default=12000)
    p.add_argument("--num-ctx", type=int, default=16384, help="fixed for all configs so only the knobs differ")
    p.add_argument("--min-similarity", type=float, default=0.98)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--json", dest="json_path", default=None, help="write full results to this file")
    p.add_argument("--dry-run", action="store_true", help="print the plan and requests, send nothing")
    p.add_argument("--yes", action="store_true", help="required to actually send requests")
    return p


def print_plan(args: argparse.Namespace, grid: list[tuple[int, int]], temps: list[float], out=print) -> None:
    per = args.repeats * 3 + 3
    out(f"server : {args.url}\nmodel  : {args.model}\nnum_ctx: {args.num_ctx}   repeats: {args.repeats}")
    out(f"configs: {len(grid)} (draft/batch, baseline first): " + ", ".join(f"{d}/{b}" for d, b in grid))
    out(f"requests per config: 1 warm-up + 1 prefill warm + {args.repeats} x 3 workloads (about {per})"
        f"; total about {per * len(grid)}" + (f" + {2 * len(temps)} determinism" if temps else ""))
    d, b = grid[0]
    for name, pl in (("W1 prefill", build_payload(args.model, prefill_messages(args.prefill_tokens, 1), d, b, args.num_ctx, 8)),
                     ("W2 code", build_payload(args.model, workload_messages("code"), d, b, args.num_ctx, 400)),
                     ("W3 json", build_payload(args.model, workload_messages("json"), d, b, args.num_ctx, 200))):
        shown = json.loads(json.dumps(pl))
        for m in shown["messages"]:
            if len(m["content"]) > 200:
                m["content"] = m["content"][:120] + f"...[{len(m['content'])} chars]"
        out(f"POST {args.url}/api/chat  {name}: {json.dumps(shown)}")


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None, out=print) -> int:
    args = make_parser().parse_args(argv)
    args.model = args.model or default_model()
    args.url = (args.url or default_url()).rstrip("/")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    grid = build_grid(parse_int_list(args.draft, "draft"), parse_int_list(args.batch, "batch"))
    temps = parse_float_list(args.temperature_list)

    print_plan(args, grid, temps, out)
    if args.dry_run:
        out("dry run: nothing was sent.")
        return 0
    if not args.yes:
        out("Nothing sent. Re-run with --yes to really send requests (the GPUs must be free).")
        return 2

    client = httpx.Client(base_url=args.url, transport=transport, timeout=httpx.Timeout(args.timeout, connect=5.0))
    try:
        ps = client.get("/api/ps")
        ps.raise_for_status()
        loaded = [m.get("name") or m.get("model") for m in ps.json().get("models", [])]
        out("loaded on the server: " + (", ".join(map(str, loaded)) if loaded else "(nothing)"))
        results = []
        for i, (d, b) in enumerate(grid, 1):
            out(f"[{i}/{len(grid)}] draft_num_predict={d} num_batch={b}")
            results.append(run_config(client, args, d, b, log=out))
        determinism = determinism_check(client, args, temps) if temps else []
    except httpx.HTTPError as exc:
        out(f"request failed: {exc}")
        return 1
    finally:
        client.close()

    verdict = analyse(results, args.min_similarity)
    out(render_table(results))
    for row in determinism:
        note = "" if row["temperature"] > 0 or row["similarity"] == 1.0 else "  NOT deterministic at temperature 0"
        out(f"determinism T={row['temperature']}: two identical W2 runs, similarity {row['similarity']:.3f}{note}")
    out(recommendation(verdict["best"], args.min_similarity))
    if args.json_path:
        best = verdict["best"]
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "url": args.url, "num_ctx": args.num_ctx, "repeats": args.repeats,
                       "min_similarity": args.min_similarity, "loaded_before": loaded, "results": results,
                       "determinism": determinism,
                       "recommendation": None if best is None else {"draft": best["draft"], "batch": best["batch"],
                                                                    "score": best["score"]}},
                      fh, indent=2)
        out(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
