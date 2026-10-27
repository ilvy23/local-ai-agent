"""Tests for agent/redact.py — secret scrubbing before log/persist."""

from agent.redact import redact


# ── SSH private keys ──────────────────────────────────────────────────────────

def test_redacts_rsa_private_key_block() -> None:
    text = (
        "Here is a key:\n"
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA3...\n"
        "-----END RSA PRIVATE KEY-----\n"
        "done"
    )
    result = redact(text)
    assert "PRIVATE KEY" not in result
    assert "MIIEpAIBAAKCAQEA3" not in result
    assert "done" in result
    assert "[REDACTED:private-key]" in result


def test_redacts_ec_private_key_block() -> None:
    text = "-----BEGIN EC PRIVATE KEY-----\nabc\n-----END EC PRIVATE KEY-----"
    assert "[REDACTED:private-key]" in redact(text)


# ── AWS access keys ───────────────────────────────────────────────────────────

def test_redacts_aws_access_key_id() -> None:
    text = "key: AKIAIOSFODNN7EXAMPLE and some other text"
    result = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in result
    assert "[REDACTED:aws-key]" in result
    assert "some other text" in result


def test_does_not_redact_short_akia_like_string() -> None:
    # "AKIA" + fewer than 16 chars — not an AWS key.
    result = redact("AKIA1234")
    assert "AKIA1234" in result


# ── Bearer tokens ─────────────────────────────────────────────────────────────

def test_redacts_bearer_token() -> None:
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc"
    result = redact(text)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
    assert "Bearer" in result
    assert "[REDACTED:token]" in result


def test_redacts_bearer_token_case_insensitive() -> None:
    text = "authorization: bearer abcdef1234567890abcdef"
    result = redact(text)
    assert "abcdef1234567890abcdef" not in result


# ── API keys ──────────────────────────────────────────────────────────────────

def test_redacts_openai_style_key() -> None:
    text = "api_key = sk-abcdefghijklmnopqrstuvwxyz123456"
    result = redact(text)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in result
    assert "[REDACTED:api-key]" in result


def test_redacts_anthropic_style_key() -> None:
    text = "key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
    result = redact(text)
    assert "sk-ant-api03-" not in result


def test_does_not_redact_short_sk_prefix() -> None:
    # "sk-" + fewer than 20 chars is not considered a key.
    result = redact("value: sk-short")
    assert "sk-short" in result


def test_redacts_github_pat() -> None:
    token = "ghp_" + "a" * 40
    result = redact(f"token: {token}")
    assert token not in result
    assert "[REDACTED:github-token]" in result


def test_redacts_github_pat_long_format() -> None:
    token = "github_pat_" + "a" * 30
    result = redact(f"auth: {token}")
    assert token not in result


# ── Database connection strings ───────────────────────────────────────────────

def test_redacts_postgresql_connection_string() -> None:
    text = "postgresql://admin:s3cr3tp@ssw0rd@db.example.com/mydb"
    result = redact(text)
    assert "s3cr3tp@ssw0rd" not in result
    assert "[REDACTED:password]" in result
    assert "db.example.com" in result


def test_redacts_mongodb_connection_string() -> None:
    text = "mongodb://user:verysecretpassword@mongo.example.com:27017/db"
    result = redact(text)
    assert "verysecretpassword" not in result


def test_does_not_touch_connection_string_without_password() -> None:
    text = "postgresql://db.example.com/mydb"
    result = redact(text)
    assert result == text


# ── Key=value assignments ─────────────────────────────────────────────────────

def test_redacts_password_assignment() -> None:
    text = 'password = "superSecretPass123!"'
    result = redact(text)
    assert "superSecretPass123!" not in result
    assert "[REDACTED]" in result


def test_redacts_api_key_assignment() -> None:
    text = "api_key=abcdef1234567890abcdef"
    result = redact(text)
    assert "abcdef1234567890abcdef" not in result


def test_does_not_redact_short_password_value() -> None:
    # Under 12 chars — not redacted.
    result = redact("password=short123")
    assert "short123" in result


# ── Safety guarantees ─────────────────────────────────────────────────────────

def test_plain_text_unchanged() -> None:
    text = "The user wants to know about Python generators."
    assert redact(text) == text


def test_empty_string() -> None:
    assert redact("") == ""


def test_never_raises_on_garbage_input() -> None:
    # Should not raise even with weird bytes-like strings or edge cases.
    result = redact("\x00\xff" + "A" * 200)
    assert isinstance(result, str)


def test_multiple_secrets_all_redacted() -> None:
    text = (
        "key1: sk-abcdefghijklmnopqrstuvwxyz123456\n"
        "key2: AKIAIOSFODNN7EXAMPLE\n"
        "pass: password=myverylongpassword123"
    )
    result = redact(text)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in result
    assert "AKIAIOSFODNN7EXAMPLE" not in result
    assert "myverylongpassword123" not in result


# ── Regression: `token` assignments ───────────────────────────────────────────

def test_redacts_token_assignment() -> None:
    """The rule's own comment lists `token`, but the alternation omitted it."""
    result = redact("token=abcdef1234567890abcdef")
    assert "abcdef1234567890abcdef" not in result
    assert "[REDACTED]" in result


def test_redacts_access_token_assignment() -> None:
    result = redact('access_token: "gho_abcdefghijklmnopqrstuvwxyz"')
    assert "gho_abcdefghijklmnopqrstuvwxyz" not in result


def test_does_not_redact_short_token_value() -> None:
    assert "abc123" in redact("token=abc123")
