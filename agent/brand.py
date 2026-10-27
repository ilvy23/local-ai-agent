"""Single source of truth for the product name. Change it here and every display string follows.

Only display text and the command name live here; the Python package (`agent`), config dirs and DB paths keep
their old names on purpose (renaming them would break installs and plugins)."""

from __future__ import annotations

NAME = "Nightage"
CMD = "nightage"
TAGLINE = "A local AI agent for chat, code and research."
ACCENT = "#8fa8d8"  # one calm colour; the UI palette is derived from it
LEGACY_CMDS = ("agent", "lantern")
AUTHOR = "ilvy23"
LICENSE_NAME = "Apache-2.0"
COPYRIGHT = f"Copyright 2026 {AUTHOR}"    # keep in sync with NOTICE (Apache-2.0 section 4 asks derived works to keep it)
# TODO: final repository name/URL is undecided; keep the current one until it is.
REPO_URL = "https://github.com/ilvy23/local-ai-agent"


def wordmark() -> str:
    """Plain one-line wordmark (no ASCII art)."""
    return NAME.lower()
