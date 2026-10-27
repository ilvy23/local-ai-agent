"""Secret redaction: scrub sensitive strings from text before logging or persisting.

Applied to tool inputs and outputs before they reach trajectory.jsonl.  Only
patterns with a clear signal-to-noise advantage are included — overly broad
patterns create false positives that mangle legitimate content and destroy
utility.  Better to miss an unusual format than to break a real file path.
"""

from __future__ import annotations

import re

# Each entry is (compiled_pattern, replacement_string).  Patterns with
# capture groups use backreferences in the replacement to preserve surrounding
# structure (e.g. keep the key name, redact only the value).
_RULES: list[tuple[re.Pattern[str], str]] = [
    # SSH private key blocks — the most unambiguous form.
    (
        re.compile(
            r"-----BEGIN\s+(?:\w+\s+)?PRIVATE KEY-----.*?-----END\s+(?:\w+\s+)?PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED:private-key]",
    ),
    # AWS access key IDs — exact 20-char format AKIA + 16 uppercase alnum.
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED:aws-key]",
    ),
    # Bearer / Authorization header tokens.
    (
        re.compile(r"(Bearer\s+)[A-Za-z0-9\-_.~+/=]{16,}", re.IGNORECASE),
        r"\1[REDACTED:token]",
    ),
    # OpenAI / Anthropic style secret keys: sk-... with 20+ chars after the prefix.
    (
        re.compile(r"\bsk-[A-Za-z0-9\-_]{20,}"),
        "[REDACTED:api-key]",
    ),
    # GitHub personal access tokens.
    (
        re.compile(r"\bghp_[A-Za-z0-9]{36,}"),
        "[REDACTED:github-token]",
    ),
    (
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"),
        "[REDACTED:github-token]",
    ),
    # Database / service connection strings with embedded credentials.
    # Matches protocol://user:password@host[/path].
    (
        re.compile(
            r"((?:postgresql|mysql|mongodb|redis|amqp|mssql)://[^:@\s]+:)"
            r"[^@\s]{1,}(@[^\s\"']+)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED:password]\2",
    ),
    # password / secret / token / api_key in key=value or key: value assignments.
    # Requires the value to look like a real secret: 12+ chars, not a plain word.
    # `token` is deliberately unanchored on the left so access_token / auth_token /
    # refresh_token are covered by the same rule.
    (
        re.compile(
            r"((?:password|passwd|secret|token|api_key|apikey|access_key|private_key)"
            r"\s*[=:]\s*[\"']?)"
            r"[A-Za-z0-9\-_.~+/=!@#$%^&*]{12,}"
            r"([\"']?)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]\2",
    ),
]


def redact(text: str) -> str:
    """Apply all redaction rules to `text` and return the sanitised result.

    Never raises — if a rule fails for any reason, the original text is left
    unchanged for that rule so the rest of the pipeline can continue.
    """
    for pattern, replacement in _RULES:
        try:
            text = pattern.sub(replacement, text)
        except Exception:  # noqa: BLE001 - redact must never break a turn
            pass
    return text
