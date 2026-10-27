#!/bin/sh
# Nightage installer for Termux (Android) - EXPERIMENTAL, not an official platform.
#
# UNVERIFIED on a real phone: this was written without an Android device. Package names (python, git, ripgrep, ruff, ollama) were
# checked against the termux-packages repository on 2026-10-26 (ollama: packages/ollama/build.sh, version 0.40.11, aarch64/x86_64
# only), everything else (pip behaviour, speed, memory) is expected, not observed.
#
# What it does (idempotent: safe to run again, it only does what is missing; no root needed):
#   1. checks it runs inside Termux (`pkg` exists) and refuses otherwise,
#   2. `pkg install` python, git, ripgrep  (no rust, no compiler: only pure-Python dependencies are installed),
#   3. makes a virtualenv in $NIGHTAGE_VENV (default ~/.nightage-venv) and installs ONLY the light dependencies
#      (typer, rich, pyyaml, httpx, prompt_toolkit). sqlite-vec / ruff / coverage are skipped on purpose:
#      memory falls back to keyword recall, the code agent skips lint and the coverage gate,
#   4. writes a launcher `nightage` into $PREFIX/bin that runs the checkout directly (the build backend of pyproject.toml needs a
#      compiled wheel Termux does not have, so the project is not pip-installed),
#   5. optionally (asks) points nightage at an Ollama on another machine (`ollama.host`, e.g. your PC over Tailscale),
#   6. optionally (asks, default no) installs Termux's own `ollama` package for fully on-phone models.
#
#   sh install-termux.sh              interactive
#   sh install-termux.sh --yes        no questions (skips the optional gateway/ollama steps)
#   sh install-termux.sh --dry-run    print what would be done, change nothing
#   NIGHTAGE_REMOTE_HOST=100.64.0.5   non-interactive remote gateway (host or host:port)
set -eu

ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes) ASSUME_YES=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

say()  { printf '%s\n' "$*"; }
warn() { printf '! %s\n' "$*" >&2; }
die()  { printf 'x %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# run CMD...  -> executes, or only prints with --dry-run
run() {
  if [ "$DRY_RUN" = 1 ]; then
    say "  [dry-run] $*"
  else
    "$@"
  fi
}

interactive() { [ "$ASSUME_YES" = 0 ] && [ -t 0 ] && [ "$DRY_RUN" = 0 ]; }

# ask "question" default(y|n)  -> 0 = yes
ask() {
  if ! interactive; then
    [ "$2" = y ] && [ "$ASSUME_YES" = 1 ]
    return
  fi
  if [ "$2" = y ]; then hint='[Y/n]'; else hint='[y/N]'; fi
  printf '? %s %s ' "$1" "$hint"
  read -r reply || reply=''
  case "$reply" in
    [yY]*) return 0 ;;
    [nN]*) return 1 ;;
    *) [ "$2" = y ] ;;
  esac
}

SRC=$(cd "$(dirname "$0")" && pwd)
VENV=${NIGHTAGE_VENV:-"$HOME/.nightage-venv"}
PREFIX_DIR=${PREFIX:-}
BIN_DIR=${PREFIX_DIR:+$PREFIX_DIR/bin}
BIN_DIR=${BIN_DIR:-"$HOME/.local/bin"}

# ---- 1. Termux? -------------------------------------------------------------------------------------------------------------
say "Nightage for Termux (experimental)"
if ! have pkg; then
  [ "$DRY_RUN" = 1 ] || die "this script is for Termux (no \`pkg\` command found). On a normal Linux use ./install.sh."
  warn "no \`pkg\` here: dry-run continues anyway"
fi
case "$PREFIX_DIR" in
  *com.termux*) ;;
  *) warn "PREFIX does not look like Termux ($PREFIX_DIR); continuing, but this is untested here" ;;
esac
[ -f "$SRC/pyproject.toml" ] || die "run this script from a nightage checkout (pyproject.toml not found next to it)."

# ---- 2. packages ------------------------------------------------------------------------------------------------------------
say "[1/5] system packages"
missing=''
for p in python git rg; do
  have "$p" || missing="$missing $p"
done
if [ -n "$missing" ]; then
  pk=''
  for p in $missing; do
    case "$p" in rg) pk="$pk ripgrep" ;; *) pk="$pk $p" ;; esac
  done
  # shellcheck disable=SC2086  # word splitting of the package list is intended
  run pkg install -y $pk
else
  say "  python, git, ripgrep present"
fi
if ! have ruff; then
  say "  tip: \`pkg install ruff\` gives the code agent its lint check (optional)"
fi

# ---- 3. virtualenv + light dependencies -------------------------------------------------------------------------------------
say "[2/5] python environment in $VENV"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  say "  exists"
else
  if [ -d "$VENV" ]; then
    say "  found a broken environment (no pip): recreating it"
    run rm -rf "$VENV"
  fi
  run python -m venv "$VENV" || die "could not create the virtualenv (Termux: \`pkg install python\` includes venv and pip)"
fi
say "[3/5] light dependencies (pure Python; no sqlite-vec / ruff / coverage)"
# Keep in step with [project].dependencies in pyproject.toml, minus the optional trio.
run "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check \
  "typer>=0.15" "rich>=13.9" "pyyaml>=6.0" "httpx>=0.28" "prompt_toolkit>=3.0.43"

# ---- 4. launcher ------------------------------------------------------------------------------------------------------------
say "[4/5] launcher $BIN_DIR/nightage"
if [ "$DRY_RUN" = 1 ]; then
  say "  [dry-run] write $BIN_DIR/nightage -> $VENV/bin/python -m agent.main"
else
  mkdir -p "$BIN_DIR"
  tmp="$BIN_DIR/.nightage.$$"
  {
    printf '#!/bin/sh\n'
    printf '# generated by install-termux.sh (re-run it to refresh)\n'
    printf 'PYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}" exec "%s/bin/python" -m agent.main "$@"\n' "$SRC" "$VENV"
  } >"$tmp"
  chmod 755 "$tmp"
  mv "$tmp" "$BIN_DIR/nightage"
fi

# ---- 5. where does the model run? -------------------------------------------------------------------------------------------
say "[5/5] model server"
remote=${NIGHTAGE_REMOTE_HOST:-}
if [ -z "$remote" ] && ask "Use Ollama on another machine (e.g. your PC over Tailscale) instead of on the phone?" n; then
  printf '  host or host:port of that Ollama (e.g. 100.64.0.5): '
  read -r remote || remote=''
fi
if [ -n "$remote" ]; then
  say "  remote gateway: $remote"
  CFG="$SRC/config.yaml"
  if [ "$DRY_RUN" = 1 ]; then
    say "  [dry-run] set ollama.host: $remote in $CFG"
  else
    NIGHTAGE_CFG="$CFG" NIGHTAGE_HOST="$remote" "$VENV/bin/python" - <<'PY'
import os, pathlib, yaml
path = pathlib.Path(os.environ["NIGHTAGE_CFG"])
data = yaml.safe_load(path.read_text()) if path.exists() else {}
data = data if isinstance(data, dict) else {}
block = data.get("ollama") if isinstance(data.get("ollama"), dict) else {}
if block.get("host"):
    print("  ollama.host already set (%s): left alone" % block["host"])
else:
    block["host"] = os.environ["NIGHTAGE_HOST"]
    data["ollama"] = block
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    print("  wrote ollama.host to", path)
PY
  fi
  say "  On the PC: Ollama must listen on the network (OLLAMA_HOST=0.0.0.0) and Tailscale must be up on both devices."
elif ask "Install Termux's own ollama package for models on the phone itself? (needs several GB free; aarch64 only)" n; then
  run pkg install -y ollama
  say "  start it with:  ollama serve &   then:  ollama pull qwen3.5:2b"
else
  say "  skipped: set ollama.host later in $SRC/config.yaml, or install ollama with \`pkg install ollama\`"
fi

say ""
say "Done. Next steps:"
say "  1. make sure $BIN_DIR is on your PATH (Termux: it is, for \$PREFIX/bin)"
say "  2. nightage --profile phone doctor        check the setup (RAM, models, remote latency)"
say "  3. nightage --profile phone               start chatting"
say "  Optional: termux-setup-storage            lets nightage read Downloads/Documents under ~/storage"
say "  More: docs/TERMUX.md, docs/PORTABILITY.md"
