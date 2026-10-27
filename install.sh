#!/bin/sh
# Nightage guided installer - Linux, macOS and Termux (POSIX sh; no bash needed).
#
# Easiest way (one line, then just answer the questions - every question has a safe default):
#     curl -LsSf <INSTALL_URL>/install.sh | sh        (see docs/INSTALL.md for the exact line)
# Careful way (download, read, then run) - recommended if you like to see what runs:
#     curl -LsSf <INSTALL_URL>/install.sh -o install.sh
#     less install.sh
#     sh install.sh
#
# What it does, in plain words (it tells you again, step by step, and asks before each change):
#   1. checks your system (OS, memory, graphics card)
#   2. installs `uv` (a small official tool from astral.sh) so YOU never have to install Python by hand
#   3. installs Nightage as an isolated tool: its own private Python, nothing mixed into your system
#   4. helps you install Ollama (the program that runs the AI model on your computer)
#   5. picks ONE model that fits this computer and offers to download it
#   6. runs `nightage doctor` and tells you what to type next
# Everything stays in your home folder. No sudo, no root, no admin. Safe to run again (repair / upgrade).
#
# Options:
#   -y, --yes          accept the safe default for every question (never runs sudo, never deletes your data)
#   -n, --dry-run      show everything that would happen, change nothing
#       --no-ollama    do not touch Ollama (use this if it runs on another machine)
#       --no-model     do not offer a model download
#       --model NAME   use this Ollama model instead of the recommended one
#       --embed TAG    use this memory (embedding) model, e.g. all-minilm; without it you choose from a short list
#       --no-embed     do not offer a memory model (Nightage then recalls by keywords only)
#       --with-ollama  say yes to installing Ollama with its official installer (Linux: this needs sudo)
#       --source X     install from X: a folder, a .tar.gz/.zip URL, or a git+https URL (default: see below)
#       --uninstall    remove Nightage (asks before touching your data)
#       --force        allow running as root (not recommended)
#   -h, --help         this text
# Environment: NIGHTAGE_REPO (project page URL), NIGHTAGE_PYTHON (default 3.12), NIGHTAGE_INSTALL_LOG, NO_COLOR.
#
# Log for bug reports: ${XDG_STATE_HOME:-~/.local/state}/nightage/install.log (the path is printed).

# Project page; must equal agent/brand.py REPO_URL (a test checks that). While it is still the bare placeholder the installer
# needs --source, NIGHTAGE_REPO, or to be run from a downloaded copy of the project.
REPO_URL="https://github.com/ilvy23/local-ai-agent"

UV_URL="https://astral.sh/uv/install.sh"
OLLAMA_SH_URL="https://ollama.com/install.sh"
OLLAMA_DL_URL="https://ollama.com/download"
TOOL_NAME="agent"            # the Python package name; it provides the commands nightage, lantern and agent
YES=0
DRY=0
NO_OLLAMA=0
NO_MODEL=0
WITH_OLLAMA=0
FORCE=0
UNINSTALL=0
SOURCE=""
MODEL_OVERRIDE=""
EMB_READY=0
EMBED_OVERRIDE=""
NO_EMBED=0
STEP=0
TOTAL=7
TMPD=""

# ---------------------------------------------------------------- looks
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
  C=$(printf '\033[36m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m')
  D=$(printf '\033[2m'); B=$(printf '\033[1m'); N=$(printf '\033[0m')
else
  C=''; G=''; Y=''; R=''; D=''; B=''; N=''
fi

ORIG_PATH="$PATH"
STATE_BASE="${XDG_STATE_HOME:-$HOME/.local/state}"
LOG="${NIGHTAGE_INSTALL_LOG:-$STATE_BASE/nightage/install.log}"

log() { # log text   (append to the log file; never fails the installer)
  [ "$DRY" = 1 ] && return 0
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null)" "$*" >>"$LOG" 2>/dev/null || true
}
say()  { printf '%s\n' "$*"; log "$*"; }
blank() { printf '\n'; }
step() { STEP=$((STEP + 1)); printf '\n%s[%s/%s]%s %s%s%s\n' "$C" "$STEP" "$TOTAL" "$N" "$B" "$1" "$N"; log "STEP $STEP: $1"; }
ok()   { printf '   %s+%s %s\n' "$G" "$N" "$1"; log "ok: $1"; }
info() { printf '   %s|%s %s\n' "$D" "$N" "$1"; log "info: $1"; }
warn() { printf '   %s!%s %s\n' "$Y" "$N" "$1"; log "warn: $1"; }
die()  { printf '   %sx%s %s\n' "$R" "$N" "$1" >&2; log "FATAL: $1"; [ "$DRY" = 1 ] || printf '   Log: %s\n' "$LOG" >&2; cleanup; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
cleanup() { if [ -n "$TMPD" ] && [ -d "$TMPD" ]; then rm -rf "$TMPD"; fi; TMPD=""; }
trap cleanup EXIT INT TERM

usage() { awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"; }

# ---------------------------------------------------------------- questions
# ask "question" y|n  -> 0 = yes, 1 = no. One question at a time; the default is what Enter (or --yes) picks.
ask() {
  ask_q="$1"; ask_d="${2:-y}"
  if [ "$ask_d" = y ]; then ask_hint="[Y/n]"; else ask_hint="[y/N]"; fi
  if [ "$YES" = 1 ] || [ "$DRY" = 1 ]; then
    if [ "$ask_d" = y ]; then ask_a="yes"; else ask_a="no"; fi
    if [ "$DRY" = 1 ] && [ "$YES" != 1 ]; then ask_note="dry run: would ask, default"; else ask_note="auto"; fi
    printf '   %s?%s %s %s[%s: %s]%s\n' "$C" "$N" "$ask_q" "$D" "$ask_note" "$ask_a" "$N"
    log "question: $ask_q -> $ask_a ($ask_note)"
    [ "$ask_d" = y ]; return
  fi
  printf '   %s?%s %s %s%s%s ' "$C" "$N" "$ask_q" "$D" "$ask_hint" "$N"
  if [ -t 0 ]; then
    read -r ask_r || ask_r=""
  elif [ -r /dev/tty ] && (: </dev/tty) 2>/dev/null; then
    read -r ask_r </dev/tty || ask_r=""
  else
    printf '\n'
    die "There is no terminal to ask you on. Run again with --yes to accept the defaults, or download the script and run it: sh install.sh"
  fi
  log "question: $ask_q -> '$ask_r'"
  case "$ask_r" in
    [yY]*) return 0 ;;
    [nN]*) return 1 ;;
    *) [ "$ask_d" = y ] ;;
  esac
}

# The memory (embedding) models offered below. MIRRORS agent/embedmodels.py (tests/test_installers.py checks that tags and
# sizes agree). Lines: number|tag|download MB|who it is for. Sizes are the ollama.com download sizes.
EMB_TABLE='1|all-minilm|46|English only - smallest, fastest
2|nomic-embed-text|274|English only - small, good all-rounder
3|embeddinggemma:300m-qat-q4_0|239|Multilingual (100+ languages, e.g. German, French, Spanish) - small
4|granite-embedding:278m|563|Multilingual - mid size
5|bge-m3|1200|Multilingual (100+ languages, e.g. German, French) - best for non-English chats, biggest'

# emb_lang -> english | multilingual | unknown, from the locale (the same rule as agent/embedmodels.py language_hint)
emb_lang() {
  el_v="${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}"
  case "$el_v" in
    ""|C|POSIX|C.*|POSIX.*) echo unknown ;;
    en|en_*|en.*|en@*|en-*) echo english ;;
    *) echo multilingual ;;
  esac
}

# emb_recommend RAM_MB LANG -> the tag to start with (the same rules as agent/embedmodels.py recommend)
emb_recommend() {
  if [ "$2" = english ]; then
    if [ "$1" -lt 4096 ]; then echo all-minilm; else echo nomic-embed-text; fi
  elif [ "$1" -lt 4096 ]; then
    if [ "$2" = unknown ]; then echo all-minilm; else echo "embeddinggemma:300m-qat-q4_0"; fi
  elif [ "$2" = multilingual ] && [ "$1" -ge 16384 ]; then echo bge-m3
  else echo "embeddinggemma:300m-qat-q4_0"; fi
}

# emb_field TAG N -> field N (2 = tag, 3 = MB, 4 = label) of the table line for TAG, empty when unknown
emb_field() {
  printf '%s\n' "$EMB_TABLE" | awk -F'|' -v t="$1" -v n="$2" '$2 == t { print $n }'
}

# choose_embed -> sets EMB to a tag, or "" for none. A numbered choice; Enter (or --yes) takes the recommended one.
choose_embed() {
  EMB=""
  ce_rec=$(emb_recommend "$RAM" "$(emb_lang)")
  if [ -n "$EMBED_OVERRIDE" ]; then EMB="$EMBED_OVERRIDE"; info "Using the memory model you asked for: $EMB"; return 0; fi
  printf '   %sThe memory model lets Nightage recall things by meaning. It runs on the CPU, so a smaller one leaves more memory for the chat model.%s\n' "$D" "$N"
  printf '%s\n' "$EMB_TABLE" | while IFS='|' read -r ce_n ce_t ce_mb ce_l; do
    ce_mark=""; [ "$ce_t" = "$ce_rec" ] && ce_mark="   <- recommended for this computer"
    printf '     %s) %-32s %8s   %s%s\n' "$ce_n" "$ce_t" "$(human_size "$ce_mb")" "$ce_l" "$ce_mark"
  done
  printf '     0) none - Nightage still works and searches memory by keywords\n'
  if [ "$YES" = 1 ] || [ "$DRY" = 1 ]; then
    if [ "$DRY" = 1 ] && [ "$YES" != 1 ]; then ce_note="dry run: would ask, default"; else ce_note="auto"; fi
    printf '   %s?%s Which memory model? %s[%s: %s]%s\n' "$C" "$N" "$D" "$ce_note" "$ce_rec" "$N"
    log "question: memory model -> $ce_rec ($ce_note)"
    EMB="$ce_rec"; return 0
  fi
  printf '   %s?%s Which memory model? %s[number, Enter = recommended]%s ' "$C" "$N" "$D" "$N"
  if [ -t 0 ]; then
    read -r ce_r || ce_r=""
  elif [ -r /dev/tty ] && (: </dev/tty) 2>/dev/null; then
    read -r ce_r </dev/tty || ce_r=""
  else
    printf '\n'
    die "There is no terminal to ask you on. Run again with --yes to accept the defaults, or download the script and run it: sh install.sh"
  fi
  log "question: memory model -> '$ce_r'"
  case "$ce_r" in
    0|[nN]*) EMB="" ;;
    [1-9]) EMB=$(printf '%s\n' "$EMB_TABLE" | awk -F'|' -v n="$ce_r" '$1 == n { print $2 }'); [ -n "$EMB" ] || EMB="$ce_rec" ;;
    *) EMB="$ce_rec" ;;
  esac
}

# run CMD ARGS...  -> shows the exact command first; in a dry run it is only shown.
run() {
  printf '   %s$%s %s\n' "$D" "$N" "$*"
  log "run: $*"
  if [ "$DRY" = 1 ]; then printf '   %s  (dry run: not executed)%s\n' "$D" "$N"; return 0; fi
  "$@"
  run_rc=$?
  log "exit code: $run_rc"
  return "$run_rc"
}

# ---------------------------------------------------------------- arguments
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) YES=1 ;;
    -n|--dry-run) DRY=1 ;;
    --no-ollama) NO_OLLAMA=1 ;;
    --no-model) NO_MODEL=1 ;;
    --with-ollama) WITH_OLLAMA=1 ;;
    --force) FORCE=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --source) [ $# -ge 2 ] || { echo "--source needs a value" >&2; exit 2; }; SOURCE="$2"; shift ;;
    --source=*) SOURCE="${1#--source=}" ;;
    --model) [ $# -ge 2 ] || { echo "--model needs a value" >&2; exit 2; }; MODEL_OVERRIDE="$2"; shift ;;
    --model=*) MODEL_OVERRIDE="${1#--model=}" ;;
    --embed) [ $# -ge 2 ] || { echo "--embed needs a value" >&2; exit 2; }; EMBED_OVERRIDE="$2"; shift ;;
    --embed=*) EMBED_OVERRIDE="${1#--embed=}" ;;
    --no-embed) NO_EMBED=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\nTry: sh install.sh --help\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- safety: not as root
if [ "$(id -u 2>/dev/null || echo 1)" = 0 ] && [ "$FORCE" != 1 ]; then
  printf '%sx%s This installer is meant for a normal user account, not root/sudo.\n' "$R" "$N" >&2
  printf '  Everything it does lives in your home folder. Run it again without sudo,\n  or add --force if you really know what you are doing.\n' >&2
  exit 1
fi

if [ "$DRY" != 1 ]; then
  mkdir -p "$(dirname "$LOG")" "$STATE_BASE/nightage" 2>/dev/null || LOG="/dev/null"
fi
TMPD=""
if [ "$DRY" != 1 ]; then TMPD=$(mktemp -d 2>/dev/null || mktemp -d -t nightage) || die "could not make a temporary folder"; fi

# ---------------------------------------------------------------- small helpers
fetch_to() { # fetch_to URL FILE   (HTTPS only; curl or wget)
  case "$1" in https://*) ;; *) die "refusing to download from a non-HTTPS address: $1" ;; esac
  if have wget && ! have curl; then run wget -q --https-only "$1" -O "$2"
  elif have curl || [ "$DRY" = 1 ]; then run curl --proto '=https' --tlsv1.2 -fLsS "$1" -o "$2"
  else die "need curl or wget to download. Install one with your package manager, then run this again."; fi
}
sha_of() { if have sha256sum; then sha256sum "$1" | awk '{print $1}'; elif have shasum; then shasum -a 256 "$1" | awk '{print $1}'; else echo "(no sha256 tool)"; fi; }
mb_free_home() { df -Pk "$HOME" 2>/dev/null | awk 'NR==2{print int($4/1024)}'; }

is_placeholder() { # a project URL with no owner/repo part is still the placeholder
  case "$1" in
    https://github.com/?*/?*) return 1 ;;
    *) return 0 ;;
  esac
}

script_dir() {
  case "$0" in
    */install.sh|install.sh) (cd "$(dirname "$0")" 2>/dev/null && pwd) ;;
    *) echo "" ;;
  esac
}

# ---------------------------------------------------------------- system facts
OS=$(uname -s 2>/dev/null || echo unknown)
ARCH=$(uname -m 2>/dev/null || echo unknown)
TERMUX=0
case "${PREFIX:-}" in *com.termux*) TERMUX=1 ;; esac
[ -n "${TERMUX_VERSION:-}" ] && TERMUX=1
case "$OS" in
  Darwin) PLATFORM="macOS" ;;
  Linux) if [ "$TERMUX" = 1 ]; then PLATFORM="Termux (Android)"; else PLATFORM="Linux"; fi ;;
  *) PLATFORM="$OS" ;;
esac

ram_mb() {
  if [ -n "${NIGHTAGE_TEST_RAM_MB:-}" ]; then echo "$NIGHTAGE_TEST_RAM_MB"; return; fi    # test hook: pretend this much RAM
  case "$OS" in
    Darwin) sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1048576)}' ;;
    *) awk '/^MemTotal:/{print int($2/1024)}' /proc/meminfo 2>/dev/null ;;
  esac
}
gpu_text() { # sets GPU_DESC (a short description) and GPU_VRAM_MB; no subshell so both survive
  GPU_VRAM_MB=0; GPU_DESC=""
  if [ -n "${NIGHTAGE_TEST_VRAM_MB:-}" ]; then GPU_VRAM_MB="$NIGHTAGE_TEST_VRAM_MB"; GPU_DESC="(test) graphics card with $GPU_VRAM_MB MB"; return 0; fi
  if have nvidia-smi; then
    gt_line=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    if [ -n "$gt_line" ]; then
      GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print int(s)}')
      GPU_DESC="NVIDIA: $gt_line MB (all cards: ${GPU_VRAM_MB} MB)"; return 0
    fi
  fi
  if [ "$OS" = Darwin ] && [ "$ARCH" = arm64 ]; then GPU_DESC="Apple Silicon (memory is shared with the CPU)"; return 0; fi
  if have lspci; then
    gt_line=$(lspci 2>/dev/null | grep -iE 'vga|3d controller' | head -n 1 | sed 's/^[^ ]* //')
    [ -n "$gt_line" ] && { GPU_DESC="$gt_line"; return 0; }
  fi
  GPU_DESC="no dedicated graphics card detected (the model will run on the CPU)"
}

# The same tiers as `nightage doctor --deep` (agent/doctor/checks/deep.py): GPU memory x0.9, else half of the RAM.
# sets PM_MODEL, PM_SIZE (MB), PM_BUDGET (MB) and PM_SRC (no subshell, so the globals survive)
pick_model() { # pick_model RAM_MB VRAM_MB
  if [ "$2" -gt 0 ]; then pm_budget=$(( $2 * 9 / 10 )); pm_src="graphics memory"; else pm_budget=$(( $1 / 2 )); pm_src="half of the RAM"; fi
  [ "$1" -le 0 ] && [ "$2" -le 0 ] && { pm_budget=4096; pm_src="unknown memory (assuming 4 GB)"; }
  PM_SRC="$pm_src"; PM_BUDGET="$pm_budget"
  if   [ "$pm_budget" -lt 3072 ];  then PM_MODEL="qwen3.5:0.8b"; PM_SIZE=1300
  elif [ "$pm_budget" -lt 8192 ];  then PM_MODEL="qwen3.5:4b"; PM_SIZE=4000
  elif [ "$pm_budget" -lt 20480 ]; then PM_MODEL="qwen3.5:9b"; PM_SIZE=7600
  else PM_MODEL="qwen3.5:27b"; PM_SIZE=20000; fi
}

human_size() { LC_ALL=C awk -v m="$1" 'BEGIN{ if (m>=1024) printf "%.1f GB", m/1024; else printf "%d MB", m }'; }

# ---------------------------------------------------------------- uninstall
data_dirs() {
  for dd in "${NIGHTAGE_HOME:-$HOME/.local/share/nightage}" "$STATE_BASE/nightage" "$STATE_BASE/agent" \
            "${XDG_CONFIG_HOME:-$HOME/.config}/agent"; do
    [ -d "$dd" ] && echo "$dd"
  done
}

do_uninstall() {
  printf '\n%s Nightage uninstaller%s\n' "$B" "$N"
  info "This removes the Nightage program. It never removes uv, Python or Ollama (how to do that is printed at the end)."
  UVBIN=""
  have uv && UVBIN=uv
  [ -z "$UVBIN" ] && [ -x "$HOME/.local/bin/uv" ] && UVBIN="$HOME/.local/bin/uv"
  if [ -n "$UVBIN" ] && "$UVBIN" tool list 2>/dev/null | grep -q "^$TOOL_NAME "; then
    ask "Remove the Nightage program (uv tool uninstall $TOOL_NAME)?" y && run "$UVBIN" tool uninstall "$TOOL_NAME"
  else
    info "No uv-installed Nightage was found."
  fi
  dirs=$(data_dirs)
  if [ -n "$dirs" ]; then
    blank; info "Your Nightage data (settings, memory database, history, plugins) lives here:"
    for dd in $dirs; do info "  $dd  ($(du -sk "$dd" 2>/dev/null | awk '{printf "%.1f MB", $1/1024}'))"; done
    warn "Deleting these cannot be undone. The default is to KEEP them."
    if ask "Delete ALL of these folders too?" n; then
      if [ "$YES" = 1 ]; then info "(--yes never deletes data; skipping)"; else
        printf '   Type the word delete to confirm: '
        read -r conf </dev/tty 2>/dev/null || conf=""
        if [ "$conf" = delete ]; then
          for dd in $dirs; do
            case "$dd" in ""|/|"$HOME"|"$HOME"/) warn "skipping unsafe path: $dd" ;;
              *) run rm -rf "$dd" ;; esac
          done
        else info "Not confirmed - nothing deleted."; fi
      fi
    else info "Your data was kept."; fi
  fi
  blank
  info "To also remove uv:     rm -f ~/.local/bin/uv ~/.local/bin/uvx   and   rm -rf \"\$(uv python dir)\"  (the Pythons uv downloaded)"
  info "To remove Ollama:      see https://docs.ollama.com (Linux: sudo rm \$(which ollama); macOS: delete the app / brew uninstall ollama)"
  info "Downloaded models:     ollama rm <name>   (list them with: ollama list)"
  ok "Done."
  exit 0
}

# ================================================================ main
printf '\n%s Nightage installer%s  %s(a local AI in your terminal; nothing leaves your computer)%s\n' "$B" "$N" "$D" "$N"
[ "$DRY" = 1 ] && printf '   %sDRY RUN: nothing will be changed, everything is only shown.%s\n' "$Y" "$N"
log "installer start: platform=$PLATFORM arch=$ARCH yes=$YES dry=$DRY args-source=$SOURCE"
[ "$UNINSTALL" = 1 ] && do_uninstall

# ---- 1. what will happen
step "What will happen"
info "This guided installer will, step by step and only after asking you:"
info "  - check your computer (read only)"
info "  - install uv (official tool from astral.sh, into ~/.local/bin) - it brings its own Python"
info "  - install Nightage in its own private environment (managed by uv, under ~/.local/share/uv)"
info "  - help you install Ollama, which runs the AI model on your computer"
info "  - download ONE AI model that fits your computer (several GB; you see the size first)"
info "It never uses sudo on its own, never edits system files, and prints every command before running it."
info "Log file (for bug reports): $LOG"
ask "Ready to start?" y || { say "Nothing was changed. Bye."; exit 0; }

# ---- 2. detect
step "Looking at your computer"
RAM=$(ram_mb); RAM=${RAM:-0}
gpu_text
ok "System: $PLATFORM, $ARCH"
if [ "$RAM" -gt 0 ]; then ok "Memory (RAM): $(human_size "$RAM")"; else warn "Could not read the amount of RAM (assuming a small computer)"; fi
ok "Graphics: $GPU_DESC"
if have python3; then info "Python already on this computer: $(python3 --version 2>&1) - not needed, uv brings its own"; else info "No Python found - that is fine, uv brings its own"; fi
FREE=$(mb_free_home); FREE=${FREE:-0}
[ "$FREE" -gt 0 ] && info "Free disk space in your home folder: $(human_size "$FREE")"
case "$ARCH" in x86_64|amd64|aarch64|arm64) ;; *) warn "Unusual processor type ($ARCH): some parts may not have ready-made downloads." ;; esac

# Termux has its own installer (no uv/compiled wheels there)
if [ "$TERMUX" = 1 ]; then
  SD=$(script_dir)
  info "Termux (Android) is installed by a separate, lighter script: install-termux.sh (EXPERIMENTAL, UNVERIFIED on real phones)."
  if [ -n "$SD" ] && [ -f "$SD/install-termux.sh" ]; then
    if ask "Run install-termux.sh now?" y; then
      set --
      [ "$YES" = 1 ] && set -- --yes
      [ "$DRY" = 1 ] && set -- "$@" --dry-run
      run sh "$SD/install-termux.sh" "$@"
      exit $?
    fi
    exit 0
  fi
  info "Get install-termux.sh from the project page (see docs/TERMUX.md), then run: sh install-termux.sh"
  exit 0
fi
case "$OS" in Linux|Darwin) ;; *) die "Unsupported system '$OS'. On Windows use install.ps1 (see docs/INSTALL.md)." ;; esac

# ---- 3. uv
step "Python, the easy way (uv)"
export PATH="$HOME/.local/bin:$PATH"
if have uv; then
  ok "uv is already installed: $(uv --version 2>/dev/null)"
else
  info "Nightage needs Python 3.11 or newer. You do not have to install it: uv (a small, widely used tool"
  info "from Astral) downloads and manages one privately for Nightage."
  info "It will download this script over HTTPS and run it as you (no root):  $UV_URL"
  info "It installs the programs uv and uvx into ~/.local/bin. Shell startup files are NOT edited by this step."
  if ask "Install uv now?" y; then
    UVSCRIPT="${TMPD:-/tmp}/uv-install.sh"
    fetch_to "$UV_URL" "$UVSCRIPT"
    if [ "$DRY" != 1 ]; then
      info "Downloaded $(wc -l <"$UVSCRIPT" | tr -d ' ') lines, sha256 $(sha_of "$UVSCRIPT")"
      info "(uv's own installer verifies the program it downloads against checksums published with each release.)"
      if ask "Read the downloaded script before it runs?" n; then "${PAGER:-less}" "$UVSCRIPT" </dev/tty 2>/dev/null || cat "$UVSCRIPT"; ask "Still run it?" y || die "Stopped at your request. Nothing else was changed."; fi
    fi
    ( export UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1; run sh "$UVSCRIPT" ) || die "The uv installer failed. See the messages above. You can also install uv by hand: https://docs.astral.sh/uv/getting-started/installation/"
    if [ "$DRY" != 1 ]; then have uv || die "uv was installed but is not found. Open a new terminal and run this installer again."; fi
    ok "uv installed"
  else
    die "uv is needed (it is how Python gets installed for you). See https://docs.astral.sh/uv/ or run again and say yes."
  fi
fi

# ---- 4. nightage
step "Installing Nightage"
SRC=""; SRC_DESC=""
if [ -n "$SOURCE" ]; then
  if [ -e "$SOURCE" ]; then SRC=$(cd "$SOURCE" 2>/dev/null && pwd || echo "$SOURCE"); [ -f "$SOURCE" ] && SRC="$SOURCE"; SRC_DESC="your --source: $SRC"
  else
    case "$SOURCE" in
      git+*) SRC="$TOOL_NAME @ $SOURCE" ;;
      https://*) SRC="$TOOL_NAME @ $SOURCE" ;;
      *) die "--source must be an existing folder/file, an https:// archive URL, or a git+https:// URL (got: $SOURCE)" ;;
    esac
    SRC_DESC="your --source: $SOURCE"
  fi
else
  REPO="${NIGHTAGE_REPO:-}"
  SD=$(script_dir)
  if [ -z "$REPO" ] && [ -n "$SD" ] && [ -f "$SD/pyproject.toml" ] && grep -q "^name = \"$TOOL_NAME\"" "$SD/pyproject.toml" 2>/dev/null; then
    SRC="$SD"; SRC_DESC="this downloaded copy of the project: $SD"
  else
    [ -z "$REPO" ] && REPO="$REPO_URL"
    if is_placeholder "$REPO"; then
      warn "The project's download address is not set in this installer yet."
      info "Tell it where Nightage is: sh install.sh --source <folder or https://... archive URL>"
      info "or set NIGHTAGE_REPO=https://github.com/<owner>/<repo>   (the project page; see docs/INSTALL.md)"
      die "No place to install Nightage from."
    fi
    case "$REPO" in https://*) ;; *) die "The project address must be https:// (got: $REPO)" ;; esac
    REPO="${REPO%/}"; REPO="${REPO%.git}"
    SRC="$TOOL_NAME @ $REPO/archive/HEAD.tar.gz"; SRC_DESC="the project page $REPO (latest version, as a plain archive: git is not needed)"
  fi
fi
info "Source: $SRC_DESC"
info "Nightage gets its own private Python environment; it cannot disturb other Python programs."
PYVER="${NIGHTAGE_PYTHON:-3.12}"
if ask "Install Nightage now?" y; then
  run uv tool install --force --python "$PYVER" "$SRC" || die "Installing Nightage failed. The messages above say why; a copy is in $LOG. Common causes: no internet, a proxy (see docs/INSTALL.md), or a wrong --source."
  ok "Nightage installed"
else
  say "Stopped before installing Nightage. Nothing else was changed."; exit 0
fi

# make sure the `nightage` command is on PATH
if [ "$DRY" = 1 ]; then BINDIR="$HOME/.local/bin"; else BINDIR=$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin"); fi
NBIN="$BINDIR/nightage"
NEWTERM=0
if [ "$DRY" != 1 ] && [ ! -x "$NBIN" ]; then warn "Expected the nightage command at $NBIN but did not find it."; fi
case ":$ORIG_PATH:" in *":$BINDIR:"*) ;; *) NEWTERM=1 ;; esac    # was it on the PATH you started with (before this script added it)?
if [ "$NEWTERM" = 1 ]; then
  warn "$BINDIR is not on your PATH yet, so typing 'nightage' would say 'command not found'."
  info "uv can add it for you: it appends one line to your shell startup file (for example ~/.bashrc, ~/.zshrc or ~/.profile) in YOUR home folder."
  if ask "Add it now (uv tool update-shell)?" y; then
    run uv tool update-shell || warn "Could not update your shell files. Add this line yourself: export PATH=\"$BINDIR:\$PATH\""
    info "Open a NEW terminal window afterwards so the change is picked up."
  else
    info "No problem. Add this line yourself to your shell startup file:  export PATH=\"$BINDIR:\$PATH\""
  fi
fi

# ---- 5. ollama
step "Ollama (the engine that runs the AI)"
OLLAMA_READY=0
if [ "$NO_OLLAMA" = 1 ]; then
  info "Skipped (--no-ollama). Nightage can use an Ollama on another computer: set ollama.host in its settings."
elif have ollama; then
  ok "Ollama is already installed: $(ollama --version 2>/dev/null | head -n 1)"; OLLAMA_READY=1
else
  warn "Ollama is not installed. It is the program that actually runs the AI model, on your own computer."
  case "$OS" in
    Darwin)
      if have brew; then
        info "Homebrew found. Official package, installs for your user:  brew install ollama"
        if ask "Install Ollama with Homebrew?" y; then run brew install ollama && OLLAMA_READY=1; fi
      else
        info "Download the official app from $OLLAMA_DL_URL (open the .dmg, drag Ollama to Applications, start it once)."
        info "Then run this installer again, it will pick up where it stopped."
      fi ;;
    Linux)
      info "Ollama's official installer for Linux is: curl -fsSL $OLLAMA_SH_URL | sh"
      warn "It needs sudo: it installs into /usr/local, creates a system user 'ollama' and a systemd service. That is a system change,"
      info "so this installer only does it if you explicitly say yes (and never with plain --yes)."
      info "Prefer to read it first?  curl -fsSL $OLLAMA_SH_URL -o ollama-install.sh && less ollama-install.sh"
      if [ "$WITH_OLLAMA" = 1 ] || ask "Download and run the official Ollama installer now (asks for your sudo password)?" n; then
        OSCRIPT="${TMPD:-/tmp}/ollama-install.sh"
        fetch_to "$OLLAMA_SH_URL" "$OSCRIPT"
        [ "$DRY" != 1 ] && info "Downloaded $(wc -l <"$OSCRIPT" | tr -d ' ') lines, sha256 $(sha_of "$OSCRIPT") (the vendor does not publish a checksum for this script)"
        run sh "$OSCRIPT" && OLLAMA_READY=1
      else
        info "Manual options, your choice:"
        info "  1) the line above, yourself, in a terminal;"
        info "  2) no sudo: download the archive for your CPU from $OLLAMA_DL_URL (ollama-linux-amd64.tar.zst or the arm64 one),"
        info "     extract it with: tar --zstd -x -C ~/.local -f <file>   then start it with: ollama serve   (UNVERIFIED path)"
        info "Then run this installer again."
      fi ;;
  esac
fi

if [ "$OLLAMA_READY" = 1 ] || have ollama; then
  if [ "$DRY" != 1 ] && ! ollama list >/dev/null 2>&1; then
    warn "The Ollama server is not answering."
    if ask "Start it now in the background (ollama serve)?" y; then
      ( nohup ollama serve >"$STATE_BASE/nightage/ollama-serve.log" 2>&1 & ) 2>/dev/null
      i=0; while [ "$i" -lt 15 ] && ! ollama list >/dev/null 2>&1; do sleep 1; i=$((i + 1)); done
      if ollama list >/dev/null 2>&1; then ok "Ollama is running"; else warn "Still not answering. Start it in another terminal: ollama serve"; fi
    fi
  fi
fi

# ---- 6. model
step "The AI model"
if [ "$NO_MODEL" = 1 ]; then
  info "Skipped (--no-model). Pick one later with: ollama pull <name>   then   nightage settings set chat_model <name>"
else
  pick_model "$RAM" "$GPU_VRAM_MB"
  MODEL="$PM_MODEL"; MSIZE="$PM_SIZE"
  [ -n "$MODEL_OVERRIDE" ] && { MODEL="$MODEL_OVERRIDE"; MSIZE=0; }
  if [ -n "$MODEL_OVERRIDE" ]; then info "Using the model you asked for: $MODEL"
  else info "For this computer (about $(human_size "$PM_BUDGET") usable for a model, from $PM_SRC) the best fit is: $MODEL  (download about $(human_size "$MSIZE"))"; fi
  if [ "$NO_OLLAMA" = 1 ] || { ! have ollama && [ "$DRY" != 1 ]; }; then
    info "Ollama is not available here, so nothing is downloaded now. Run this installer again once Ollama is installed,"
    info "or later:  ollama pull $MODEL   then   nightage settings set chat_model $MODEL"
  else
    if [ "$DRY" != 1 ] && ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxF "$MODEL"; then
      ok "$MODEL is already downloaded"
    else
      [ "$MSIZE" -gt 0 ] && [ "$FREE" -gt 0 ] && [ "$FREE" -lt $((MSIZE + 2048)) ] && warn "You have $(human_size "$FREE") free; the model needs about $(human_size "$MSIZE") plus some room."
      if ask "Download $MODEL now (one time)?" y; then
        run ollama pull "$MODEL" || warn "Download failed. Check your internet and run:  ollama pull $MODEL"
      else
        info "Skipped. Later:  ollama pull $MODEL"
      fi
    fi
    if [ "$DRY" = 1 ] || ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qxF "$MODEL"; then
      if ask "Tell Nightage to use $MODEL for chat?" y; then
        run "$NBIN" settings set chat_model "$MODEL" || warn "Could not save it now; run later:  nightage settings set chat_model $MODEL"
      fi
    fi
    # the memory model: recall by meaning instead of by keywords (Nightage works without it)
    if [ "$NO_EMBED" = 1 ]; then
      info "Skipped (--no-embed). Nightage searches memory by keywords. Later:  nightage embed list"
    else
      choose_embed
      if [ -z "$EMB" ]; then
        info "No memory model. Nightage still works and falls back to keyword search. Later:  nightage embed list"
      else
        ESIZE=$(emb_field "$EMB" 3); ESIZE_TXT=""
        [ -n "$ESIZE" ] && ESIZE_TXT=" (about $(human_size "$ESIZE"))"
        if [ "$DRY" != 1 ] && ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qE "^$EMB(:|$)"; then
          ok "$EMB (the memory model) is already downloaded"
          EMB_READY=1
        elif ask "Download the memory model $EMB${ESIZE_TXT}?" y; then
          if run ollama pull "$EMB"; then EMB_READY=1; else warn "Download failed. Later:  nightage embed use $EMB"; fi
        else
          info "Skipped. Nightage still works and falls back to keyword search. Later:  nightage embed use $EMB"
        fi
        # `embed use` also rebuilds the memory index when the model changes, so an older install never mixes vector sizes
        if [ "$EMB_READY" = 1 ] || [ "$DRY" = 1 ]; then
          run "$NBIN" embed use "$EMB" --yes --no-pull || warn "Could not switch now; run later:  nightage embed use $EMB"
        fi
      fi
    fi
  fi
fi

# ---- 7. finish
step "Checking everything"
if [ "$DRY" = 1 ]; then
  info "Dry run: would run  $NBIN doctor --no-web --only system,python,packages"
else
  "$NBIN" doctor --no-web --only system,python,packages || warn "The check reported something; the lines above say what. Run 'nightage doctor' for the full report."
fi
blank
printf '%s All done.%s\n' "$G$B" "$N"
printf '   What to type next %s(open a new terminal first if the command is not found)%s:\n' "$D" "$N"
printf '     %snightage%s          start here: the home screen\n' "$C" "$N"
printf '     %snightage demo%s     see what it looks like, no model needed\n' "$C" "$N"
printf '     %snightage chat%s     start talking\n' "$C" "$N"
blank
printf '   %sTo update later:%s nightage update       %sTo remove:%s sh install.sh --uninstall   %s(or: uv tool uninstall %s)%s\n' "$D" "$N" "$D" "$N" "$D" "$TOOL_NAME" "$N"
[ "$DRY" != 1 ] && printf '   %sInstall log: %s%s\n' "$D" "$LOG" "$N"
log "installer finished"
exit 0
