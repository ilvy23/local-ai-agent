from __future__ import annotations

import ast
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from agent.context.budget import estimate_tokens

CHAT_FILE_WEIGHT = 50.0
MENTIONED_WEIGHT = 10.0
DEFAULT_BUDGET_TOKENS = 1000
PAGERANK_DAMPING = 0.85
PAGERANK_ITERATIONS = 40


@dataclass(frozen=True)
class Tags:
    path: str
    defines: frozenset[str]
    references: Counter[str]


@dataclass
class RankedFile:
    path: str
    rank: float
    symbols: list[str] = field(default_factory=list)


def extract_tags(path: str, source: str) -> Tags:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return Tags(path=path, defines=frozenset(), references=Counter())

    defines: set[str] = set()
    references: Counter[str] = Counter()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defines.add(node.name)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defines.add(node.id)
            else:
                references[node.id] += 1
        elif isinstance(node, ast.Attribute):
            references[node.attr] += 1

    for name in defines:
        references.pop(name, None)

    return Tags(path=path, defines=frozenset(defines), references=references)


def build_tags(files: dict[str, str]) -> list[Tags]:
    return [extract_tags(path, source) for path, source in files.items()]


def _definers(tags: list[Tags]) -> dict[str, list[str]]:
    definers: dict[str, list[str]] = defaultdict(list)
    for tag in tags:
        for name in tag.defines:
            definers[name].append(tag.path)
    return definers


def rank_files(
    files: dict[str, str],
    *,
    chat_files: set[str] | None = None,
    mentioned: set[str] | None = None,
) -> list[RankedFile]:
    chat_files = chat_files or set()
    mentioned = mentioned or set()
    tags = build_tags(files)
    paths = [t.path for t in tags]
    if not paths:
        return []

    definers = _definers(tags)
    edges: dict[str, Counter[str]] = {p: Counter() for p in paths}

    for tag in tags:
        for name, count in tag.references.items():
            targets = definers.get(name)
            if not targets:
                continue
            weight = math.sqrt(count)
            if name in mentioned:
                weight *= MENTIONED_WEIGHT
            for target in targets:
                if target != tag.path:
                    edges[tag.path][target] += weight

    personalization = {
        p: (CHAT_FILE_WEIGHT if p in chat_files else 1.0) for p in paths
    }
    ranks = _pagerank(paths, edges, personalization)

    definitions = {t.path: sorted(t.defines) for t in tags}
    ranked = [
        RankedFile(path=p, rank=ranks[p], symbols=definitions.get(p, []))
        for p in paths
    ]
    ranked.sort(key=lambda r: r.rank, reverse=True)
    return ranked


def _pagerank(
    nodes: list[str],
    edges: dict[str, Counter[str]],
    personalization: dict[str, float],
) -> dict[str, float]:
    n = len(nodes)
    total_p = sum(personalization.values()) or 1.0
    teleport = {node: personalization[node] / total_p for node in nodes}
    rank = {node: 1.0 / n for node in nodes}

    out_weight = {node: sum(targets.values()) for node, targets in edges.items()}

    for _ in range(PAGERANK_ITERATIONS):
        incoming = dict.fromkeys(nodes, 0.0)
        dangling = 0.0
        for node in nodes:
            if out_weight[node] == 0:
                dangling += rank[node]
                continue
            share = rank[node] / out_weight[node]
            for target, weight in edges[node].items():
                incoming[target] += share * weight

        new_rank = {}
        for node in nodes:
            base = (1 - PAGERANK_DAMPING) * teleport[node]
            flow = PAGERANK_DAMPING * (incoming[node] + dangling * teleport[node])
            new_rank[node] = base + flow
        rank = new_rank

    return rank


def render_map(
    files: dict[str, str],
    *,
    chat_files: set[str] | None = None,
    mentioned: set[str] | None = None,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> str:
    ranked = rank_files(files, chat_files=chat_files, mentioned=mentioned)
    chat_files = chat_files or set()

    lines: list[str] = []
    used = 0
    for entry in ranked:
        if entry.path in chat_files or not entry.symbols:
            continue
        block = [f"{entry.path}:"]
        block.extend(f"  {sym}" for sym in entry.symbols)
        text = "\n".join(block)
        cost = estimate_tokens(text)
        if used + cost > budget_tokens and lines:
            break
        lines.append(text)
        used += cost
    return "\n".join(lines)
