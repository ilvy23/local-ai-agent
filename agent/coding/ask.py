"""`ask_user`: let the agent ask the human PRODUCT decisions only.

The agent must decide every code/technical question itself. A one-line LLM
classifier deflects "which library / what syntax / how do I" style questions so
the human is only ever bothered with genuine product/design forks
(colours, wording, UX, scope).
"""

from __future__ import annotations

import re

# Fast pre-filter: obvious NON-product questions never reach the user. Two families:
#   (a) code specifics — libraries/syntax/algorithms
#   (b) mechanical permission — "should I create/install/write X?", "do you want me
#       to…?", "can I…?". These are the model wanting a rubber stamp for every step
#       instead of just doing it, and they burn the whole run in the wild.
_CODE_RE = re.compile(
    r"\b(syntax|import|function|method|library|framework|algorithm|regex|"
    r"which (loop|data structure|type)|how do i (code|implement|write)|"
    r"compile|traceback|exception|dependency|package|api call|"
    # naming / structure decisions are the developer's job — never the user's
    r"what.{0,20}(name|call|filename|file name|module name|class name|"
    r"variable name|function name|directory|folder|structure|architecture))\b", re.I)
_PERMISSION_RE = re.compile(
    r"\b(should i|should we|do (you|we) want (me |us )?to|can i|can we|"
    r"shall (i|we)|may i|let'?s|let us|is it (ok|okay|alright) (if|for me) to)\b"
    r".*\b(create|make|write|add|install|use|setup|set up|configure|import|delete|remove|"
    r"run|call|move|rename|start|build|generate|initialise|initialize|include|"
    r"proceed|continue|refactor|extend|comprehensive|expand)\b",
    re.I)

_CLASSIFY = (
    "A coding agent wants to ask its human this: {q!r}\n"
    "Is it a PRODUCT/design decision only the human can make (colours, wording, "
    "layout, UX, feature scope, naming shown to end users), or a CODE/technical "
    "decision the engineer should make alone (libraries, syntax, structure, "
    "algorithms)? Answer one word: PRODUCT or CODE."
)


def is_product_question(question: str, llm=None, model: str | None = None) -> bool:
    if _CODE_RE.search(question) or _PERMISSION_RE.search(question):
        return False  # code specifics or permission-to-act — the agent must decide alone
    if llm is None or model is None:
        return True  # no classifier available → let it through
    try:
        out = "".join(llm.chat([{"role": "user", "content": _CLASSIFY.format(q=question)}],
                               model=model, temperature=0.0)).upper()
    except Exception:  # noqa: BLE001 - classifier failure shouldn't block the user
        return True
    return "PRODUCT" in out and "CODE" not in out.replace("PRODUCT", "")
