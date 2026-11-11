"""The charter: the commands a night run may use, settled in the first minutes and confirmed once by you.

`propose()` starts from what the project itself needs (profile.py: its test command, its language) and, if a model call is given, lets the model
add what the prompts imply. Every pattern is validated: a bare `*`, a pattern that starts with a wildcard, or one containing a shell operator is dropped.
`apply()` writes it into your per-repo policy (auto mode: edits and the charter's commands run without asking INSIDE the folder; the safety
classifier's BLOCKED tier, deny patterns and paths outside the folder stay refused). The previous policy of the folder is saved and
`restore_policy()` puts it back when the night ends."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.codeagent import policy as policy_mod, profile as profile_mod
from agent.jsonx import extract_json_value

_OPERATORS = re.compile(r"[;&|`<>\n]|\$\(")
BASE = {"python": ["python *", "python3 *", "pytest *", "uv run *"], "node": ["node *", "npm *", "npx *"], "rust": ["cargo *"], "go": ["go *"], "make": ["make *"]}


@dataclass
class Charter:
    allow: list[str] = field(default_factory=list)          # run without asking when not dangerous
    dangerous: list[str] = field(default_factory=list)      # run without asking although the classifier rates them DANGEROUS (inside the folder)
    network: bool = False
    installs: bool = False
    why: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)

    def show(self) -> str:
        lines = [f"allow      {', '.join(self.allow) or '-'}", f"dangerous  {', '.join(self.dangerous) or '-'}",
                 f"network    {'on' if self.network else 'off'}   installs {'on' if self.installs else 'off'}"]
        lines += [f"  {pat}: {why}" for pat, why in self.why.items()]
        return "\n".join(lines)


def valid_pattern(pat: Any) -> bool:
    p = str(pat or "").strip()
    return bool(p) and len(p) <= 80 and p[0].isalpha() and not _OPERATORS.search(p) and p.rstrip("* ") != ""


def _clean(patterns: Any) -> list[str]:
    out: list[str] = []
    for p in patterns if isinstance(patterns, list) else []:
        if valid_pattern(p) and str(p).strip() not in out:
            out.append(str(p).strip())
    return out


def baseline(root: Path) -> Charter:
    prof = profile_mod.detect(root)
    allow = [pat for kind in prof.kinds for pat in BASE.get(kind, [])]
    test = (prof.test_cmd or "").strip()
    if test and valid_pattern(test):
        allow.append(test)
    return Charter(allow=_clean(allow), why={p: "the project's own tooling" for p in allow})


def propose(root: Path, prompts: list[str], ask: Callable[[str], str] | None = None) -> Charter:
    """`ask(text) -> reply` is one bounded model call; without it (or when it fails) the baseline stands."""
    ch = baseline(root)
    if ask is None:
        return ch
    question = ("An unattended coding agent will work on these tasks in a project folder. List the shell commands it will need beyond reading and editing files, as "
                'JSON only: {"commands": [{"pattern": "make *", "why": "...", "dangerous": false}], "network": false, "installs": false}. '
                "Patterns are globs for ONE simple command (no ; & | > or $()). 'dangerous': true for commands that delete or overwrite things.\n\nTASKS:\n"
                + "\n---\n".join(p[:3000] for p in prompts))
    try:
        data = extract_json_value(ask(question), "{", "}")
    except Exception:  # noqa: BLE001 - a failed proposal leaves the baseline, never an open charter
        return ch
    for c in (data or {}).get("commands", []) if isinstance(data, dict) else []:
        if isinstance(c, dict) and valid_pattern(c.get("pattern")):
            pat = str(c["pattern"]).strip()
            target = ch.dangerous if c.get("dangerous") is True else ch.allow
            if pat not in target:
                target.append(pat)
            ch.why[pat] = str(c.get("why") or "")[:120]
    if isinstance(data, dict):
        ch.network, ch.installs = data.get("network") is True, data.get("installs") is True
    return ch


def _before_file(root: Path) -> Path:
    return root / ".night" / "policy-before.json"


def apply(root: Path, ch: Charter) -> None:
    root = root.resolve()
    f = _before_file(root)
    if not f.exists():                                      # keep the FIRST original: a resumed night must not save its own policy as the original
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps((policy_mod._load_raw().get("repos") or {}).get(str(root))))
    policy_mod.save_repo(root, mode=policy_mod.AUTO, allow=ch.allow, dangerous_allow=ch.dangerous, network=ch.network, installs=ch.installs, commit="never")


def restore_policy(root: Path) -> bool:
    root = root.resolve()
    f = _before_file(root)
    if not f.exists():
        return False
    before = json.loads(f.read_text() or "null")
    raw = policy_mod._load_raw()
    if before is None:
        policy_mod.forget_repo(root)
    else:
        raw.setdefault("repos", {})[str(root)] = before
        policy_mod.policy_file().write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    f.unlink()
    return True
