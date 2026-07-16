#!/usr/bin/env bash
# agent installer — Linux (Ubuntu/Debian, Arch, Fedora, openSUSE).
#
# Idempotent: re-running is safe, it only does what's missing. Nothing needs
# root except installing a missing package (curl) or Ollama itself.
#
#   ./install.sh          interactive
#   ./install.sh --yes    assume yes to every prompt (also the default when
#                         piped, so CI and `curl … | bash` don't hang)
set -euo pipefail

# ── looks ───────────────────────────────────────────────────────────────────
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C=$'\033[36m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'
  D=$'\033[2m'; B=$'\033[1m'; N=$'\033[0m'
else
  C=''; G=''; Y=''; R=''; D=''; B=''; N=''
fi

ASSUME_YES=0
case "${1:-}" in -y|--yes) ASSUME_YES=1 ;; esac
[ -t 0 ] || ASSUME_YES=1   # piped / no terminal: never block on a prompt

STEP=0
TOTAL=4

step()  { STEP=$((STEP + 1)); printf '\n%s[%s/%s]%s %s%s%s\n' "$C" "$STEP" "$TOTAL" "$N" "$B" "$1" "$N"; }
ok()    { printf '   %s✓%s %b\n' "$G" "$N" "$1"; }
info()  { printf '   %s│%s %b\n' "$D" "$N" "$1"; }
warn()  { printf '   %s!%s %b\n' "$Y" "$N" "$1"; }
die()   { printf '   %s✗%s %b\n' "$R" "$N" "$1" >&2; exit 1; }
have()  { command -v "$1" >/dev/null 2>&1; }
served() { curl -sf --max-time 3 http://localhost:11434/api/version >/dev/null 2>&1; }

ask() { # ask "question"  ->  0 = yes, 1 = no   (default yes)
  if [ "$ASSUME_YES" = 1 ]; then
    printf '   %s?%s %b %s[yes]%s\n' "$C" "$N" "$1" "$D" "$N"
    return 0
  fi
  printf '   %s?%s %b %s[Y/n]%s ' "$C" "$N" "$1" "$D" "$N"
  read -r reply </dev/tty || return 0
  case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

cd "$(cd "$(dirname "$0")" && pwd)"

printf '\n'
printf '%s ╭─────────────────────────────────────────────╮%s\n' "$C" "$N"
printf '%s │%s   %sagent%s — a local AI in your terminal       %s│%s\n' "$C" "$N" "$B" "$N" "$C" "$N"
printf '%s │%s   %sruns on your machine. nothing phones home%s %s│%s\n' "$C" "$N" "$D" "$N" "$C" "$N"
printf '%s ╰─────────────────────────────────────────────╯%s\n' "$C" "$N"

# ── 1. system ───────────────────────────────────────────────────────────────
step "Checking your system"
ok "$( . /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-${ID:-Linux}}" )"

if have curl; then
  ok "curl"
else
  warn "curl is missing — it's needed to download everything else"
  if ask "Install curl with your package manager? ${D}(needs sudo)${N}"; then
    if   have apt-get; then sudo apt-get update -qq && sudo apt-get install -y curl
    elif have pacman;  then sudo pacman -Sy --noconfirm curl
    elif have dnf;     then sudo dnf install -y curl
    elif have zypper;  then sudo zypper --non-interactive install curl
    else die "No package manager I recognise. Install curl, then re-run."; fi
    ok "curl installed"
  else
    die "curl is required. Install it, then re-run."
  fi
fi

# ── 2. python toolchain ─────────────────────────────────────────────────────
step "Python toolchain"
export PATH="$HOME/.local/bin:$PATH"
if have uv; then
  ok "uv $(uv --version 2>/dev/null | awk '{print $2}') ${D}already installed${N}"
else
  info "${B}uv${N} installs Python and this project's dependencies, into"
  info "your home directory — no system packages touched."
  info "source: ${C}https://astral.sh/uv/install.sh${N}"
  ask "Install uv?" || die "uv is required. See https://docs.astral.sh/uv/"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
  have uv || die "uv installed but isn't on PATH — open a new shell and re-run."
  ok "uv installed"
fi

info "installing dependencies ${D}(first run also fetches a matching Python)${N}…"
uv sync --quiet
ok "dependencies ready"

# ── 3. ollama ───────────────────────────────────────────────────────────────
step "Ollama"
if have ollama; then
  ok "Ollama $(ollama --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -1) ${D}already installed${N}"
else
  warn "Ollama is not installed"
  info "It's what actually runs the AI models, locally on this machine —"
  info "the reason none of your conversations leave your computer."
  info "official installer: ${C}https://ollama.com/install.sh${N}"
  if ask "Download and install Ollama now?"; then
    curl -fsSL https://ollama.com/install.sh | sh
    have ollama || die "Ollama installed but isn't on PATH. Open a new shell and re-run."
    ok "Ollama installed"
  else
    printf '\n'
    info "No problem — grab it yourself from ${C}https://ollama.com/download${N}"
    info "then re-run this script. It picks up where it left off."
    exit 0
  fi
fi

if served; then
  ok "server is running"
else
  info "starting the Ollama server…"
  (ollama serve >/dev/null 2>&1 &) || true
  for _ in $(seq 1 15); do served && break; sleep 1; done
  if served; then ok "server is running"
  else warn "couldn't start it — run ${C}ollama serve${N} in another terminal"; fi
fi

# ── 4. models ───────────────────────────────────────────────────────────────
step "Models"
MISSING=""
for m in "qwen2.5:7b" "bge-m3"; do
  if ollama list 2>/dev/null | grep -q "^${m%%:*}"; then
    ok "$m ${D}already downloaded${N}"
  else
    MISSING="$MISSING $m"
  fi
done

if [ -n "$MISSING" ]; then
  info "${B}qwen2.5:7b${N} ${D}(4.7 GB)${N} does the thinking — it must be a model"
  info "that supports tools, or the agent can't read files or search."
  info "${B}bge-m3${N} ${D}(1.2 GB)${N} powers the memory."
  info "missing:${B}$MISSING${N}"
  if ask "Download them now? ${D}(one time)${N}"; then
    for m in $MISSING; do
      info "pulling $m…"
      ollama pull "$m"
      ok "$m ready"
    done
  else
    warn "skipped — agent won't work until you run:"
    for m in $MISSING; do info "  ${C}ollama pull $m${N}"; done
  fi
fi

# ── done ────────────────────────────────────────────────────────────────────
printf '\n'
printf '%s ╭─────────────────────────────────────────────╮%s\n' "$G" "$N"
printf '%s │%s   %sAll set.%s                                  %s│%s\n' "$G" "$N" "$B" "$N" "$G" "$N"
printf '%s │%s                                             %s│%s\n' "$G" "$N" "$G" "$N"
printf '%s │%s     %suv run agent menu%s  %s← start here%s         %s│%s\n' "$G" "$N" "$C" "$N" "$D" "$N" "$G" "$N"
printf '%s │%s     %suv run agent chat%s  %s← straight to a chat%s %s│%s\n' "$G" "$N" "$C" "$N" "$D" "$N" "$G" "$N"
printf '%s ╰─────────────────────────────────────────────╯%s\n' "$G" "$N"
printf '\n   %sTip:%s end any message with %s/web%s to search the internet.\n\n' "$D" "$N" "$C" "$N"
