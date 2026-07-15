#!/usr/bin/env bash
# agent installer — Linux (Ubuntu/Debian, Arch, Fedora, openSUSE).
#
# Idempotent: re-running is safe. It installs uv (which fetches the right
# Python for you), syncs dependencies, ensures Ollama is installed and running,
# and pulls the default models. Nothing here needs root except installing curl
# if it's somehow missing.
set -euo pipefail

say()  { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*"; }
err()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

cd "$(cd "$(dirname "$0")" && pwd)"

DISTRO="$( . /etc/os-release 2>/dev/null && echo "${ID:-unknown}" )"
say "detected distro: $DISTRO"

# --- prerequisite: curl (via whatever package manager exists) --------------
if ! command -v curl >/dev/null 2>&1; then
  say "installing curl…"
  if   command -v apt-get >/dev/null; then sudo apt-get update && sudo apt-get install -y curl
  elif command -v pacman  >/dev/null; then sudo pacman -Sy --noconfirm curl
  elif command -v dnf     >/dev/null; then sudo dnf install -y curl
  elif command -v zypper  >/dev/null; then sudo zypper install -y curl
  else err "please install curl, then re-run this script"; exit 1; fi
fi

# --- uv: manages the Python toolchain + the virtualenv ---------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
command -v uv >/dev/null 2>&1 || { err "uv is not on PATH; open a new shell and re-run"; exit 1; }

# --- dependencies (uv downloads the pinned Python automatically) -----------
say "installing dependencies (first run also fetches Python)…"
uv sync

# --- Ollama: the local model server ----------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  say "installing Ollama…"
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
  say "starting Ollama in the background…"
  (ollama serve >/dev/null 2>&1 &) || warn "could not auto-start Ollama; run 'ollama serve' yourself"
  sleep 3
fi

# --- models (skip any already present) -------------------------------------
for m in dolphin3:8b bge-m3; do
  if ollama list 2>/dev/null | grep -q "${m%%:*}"; then
    say "model $m already present"
  else
    say "pulling $m (a few GB, one time)…"
    ollama pull "$m"
  fi
done

echo
say "done!  start agent with:"
echo "    uv run agent menu     # interactive menu"
echo "    uv run agent chat     # jump into a chat"
