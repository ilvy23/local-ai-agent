from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpawnResult:
    role: str
    output: str
    truncated: bool = False


@dataclass
class SubagentSpec:
    role: str
    system: str
    build_prompt: Any
    parse: Any = None
    max_chars: int = 3200


def spawn(
    client: Any,
    spec: SubagentSpec,
    inputs: dict[str, Any],
    *,
    model: str,
    temperature: float = 0.2,
    num_ctx: int | None = None,
) -> SpawnResult:
    prompt = spec.build_prompt(inputs)
    messages = [
        {"role": "system", "content": spec.system},
        {"role": "user", "content": prompt},
    ]
    raw = "".join(
        client.chat(
            messages=messages,
            model=model,
            stream=False,
            num_ctx=num_ctx,
            temperature=temperature,
        )
    )
    truncated = len(raw) > spec.max_chars
    output = raw[: spec.max_chars] if truncated else raw
    if spec.parse is not None:
        output = spec.parse(output)
    return SpawnResult(role=spec.role, output=output, truncated=truncated)
