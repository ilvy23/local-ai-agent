#!/bin/sh
# Nightage installer for Termux (Android) - EXPERIMENTAL, not an official platform. POSIX sh, no root.
#
# UNVERIFIED on a real phone: written without an Android device and tested only with fake `pkg`, `ollama`, `tput` and `python` programs.
# Package names (python, git, ripgrep, ollama) were checked against the termux-packages repository on 2026-10-26 (ollama 0.40.11,
# aarch64 and x86_64 only). Speed, memory and pip behaviour on your phone are expected, not observed.
#
# What it does (safe to run again: it only does what is missing). Eight short steps with a percent bar:
#   1 check the phone (Termux, memory, free space)     5 Ollama on the phone (optional): install, start, keep awake
#   2 install python, git, ripgrep                     6 download the chat model (percent bar)
#   3 Python environment with light dependencies       7 download the memory model (small)
#   4 launcher `nightage` + settings                   8 health check
# Quiet by default: pip and pkg output goes to a log file; a failing step shows the last 8 lines and the log path.
# The screen adapts to your terminal width (a phone is about 40-60 columns).
#
# Options:
#   -y, --yes          accept the default answer to every question (never deletes anything)
#   -n, --dry-run      show what would happen, change nothing
#   -v, --verbose      show the output of pkg/pip instead of only logging it
#       --no-ollama    do not install or start Ollama on the phone (use one on another machine, see below)
#       --model TAG    chat model to download, e.g. qwen3.5:2b (default: chosen by the phone's memory)
#       --embed TAG    memory (embedding) model, e.g. all-minilm (default: asked, English or other languages)
#       --no-embed     no memory model: Nightage then recalls by keywords only
#       --profile P    phone (default on Termux) or default; `phone` = small models, short context, frees memory on exit
#   -h, --help         this text
# Environment: NIGHTAGE_REMOTE_HOST=host[:port] uses the Ollama of another machine (e.g. your PC over Tailscale) and skips steps 5-7;
#              NIGHTAGE_VENV (default ~/.nightage-venv), NO_COLOR, COLUMNS.
set -eu

ASSUME_YES=0
DRY_RUN=0
VERBOSE=0
NO_OLLAMA=0
NO_EMBED=0
MODEL_OVERRIDE=''
EMBED_OVERRIDE=''
PROFILE_ARG=''

need_value() { [ "$2" -ge 2 ] || { printf '%s needs a value\n' "$1" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes) ASSUME_YES=1 ;;
    -n|--dry-run) DRY_RUN=1 ;;
    -v|--verbose) VERBOSE=1 ;;
    --no-ollama) NO_OLLAMA=1 ;;
    --no-embed) NO_EMBED=1 ;;
    --model) need_value "$1" $#; MODEL_OVERRIDE=$2; shift ;;
    --model=*) MODEL_OVERRIDE=${1#--model=} ;;
    --embed) need_value "$1" $#; EMBED_OVERRIDE=$2; shift ;;
    --embed=*) EMBED_OVERRIDE=${1#--embed=} ;;
    --profile) need_value "$1" $#; PROFILE_ARG=$2; shift ;;
    --profile=*) PROFILE_ARG=${1#--profile=} ;;
    -h|--help) awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) printf 'unknown option: %s (try --help)\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
case "$PROFILE_ARG" in ''|phone|default) ;; *) printf '%s\n' "--profile must be phone or default" >&2; exit 2 ;; esac

have() { command -v "$1" >/dev/null 2>&1; }

SRC=$(cd "$(dirname "$0")" && pwd)
VENV=${NIGHTAGE_VENV:-"$HOME/.nightage-venv"}
PREFIX_DIR=${PREFIX:-}
BIN_DIR=${PREFIX_DIR:+$PREFIX_DIR/bin}
BIN_DIR=${BIN_DIR:-"$HOME/.local/bin"}
NBIN="$BIN_DIR/nightage"
CFG="$SRC/config.yaml"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/nightage"
LOG="$STATE_DIR/install.log"
LAST="$STATE_DIR/install-last.out"      # output of the command that just ran (shown on failure; also appended to the log)

# ---------------------------------------------------------------- how the screen looks
# Width: tput -> stty -> $COLUMNS -> 40; read again before every screen (the keyboard may have been opened or the phone turned).
WIDTH=40
refresh_width() {
  w=''
  if have tput; then w=$(tput cols 2>/dev/null) || w=''; fi
  case $w in ''|*[!0-9]*) w='' ;; esac
  if [ -z "$w" ] && have stty; then
    w=$({ stty size </dev/tty; } 2>/dev/null) || w=''
    w=${w##* }
    case $w in ''|*[!0-9]*) w='' ;; esac
  fi
  if [ -z "$w" ]; then
    w=${COLUMNS:-}
    case $w in ''|*[!0-9]*) w='' ;; esac
  fi
  [ -n "$w" ] && [ "$w" -gt 0 ] || w=40
  [ "$w" -lt 24 ] && w=24
  [ "$w" -gt 72 ] && w=72                       # long lines are hard to read; the bar still grows up to 30
  WIDTH=$w
}

UNI=0
LIVE=0
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in *[uU][tT][fF]-8*|*[uU][tT][fF]8*) UNI=1 ;; esac
if [ -n "${NO_COLOR:-}" ] || [ "${TERM:-dumb}" = dumb ]; then UNI=0; fi
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != dumb ]; then LIVE=1; fi
if [ "$UNI" = 1 ]; then
  FILL='█'; EMPTY='░'; CHK='✓'; CRS='✗'; BANG='!'; BUL='•'
else
  FILL='#'; EMPTY='-'; CHK='+'; CRS='x'; BANG='!'; BUL='*'
fi
if [ "$LIVE" = 1 ]; then
  ESC=$(printf '\033')
  ACC="${ESC}[36m"; GRN="${ESC}[32m"; YEL="${ESC}[33m"; RED="${ESC}[31m"; DIM="${ESC}[2m"; BLD="${ESC}[1m"; RST="${ESC}[0m"
else
  ESC=''; ACC=''; GRN=''; YEL=''; RED=''; DIM=''; BLD=''; RST=''
fi

# wrap FIRST_PREFIX REST_INDENT TEXT : words never split unless one is longer than the line; no line longer than WIDTH-1.
# The prefix may carry colour codes; its VISIBLE width must equal the length of REST_INDENT (spaces).
wrap() {
  printf '%s\n' "$3" | awk -v w="$((WIDTH - 1))" -v p1="$1" -v pr="$2" '
    function emit(s) { print (n == 0 ? p1 : pr) s; n++ }
    BEGIN { avail = w - length(pr); if (avail < 8) avail = 8; line = ""; n = 0 }
    { for (i = 1; i <= NF; i++) {
        word = $i
        if (length(word) > avail) {
          if (line != "") { emit(line); line = "" }
          while (length(word) > avail) { emit(substr(word, 1, avail)); word = substr(word, avail + 1) }
        }
        if (line == "") line = word
        else if (length(line) + 1 + length(word) <= avail) line = line " " word
        else { emit(line); line = word }
    } }
    END { if (line != "" || n == 0) emit(line) }'
}

log() {
  [ "$DRY_RUN" = 1 ] && return 0
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo -)" "$*" >>"$LOG" 2>/dev/null || true
}
say()  { wrap '  ' '  ' "$*"; log "$*"; }
good() { wrap "  ${GRN}${CHK}${RST} " '    ' "$*"; log "ok: $*"; }
note() { wrap "  ${DIM}${BUL}${RST} " '    ' "$*"; log "note: $*"; }
warn() { wrap "  ${YEL}${BANG}${RST} " '    ' "$*"; log "warn: $*"; }
bad()  { wrap "  ${RED}${CRS}${RST} " '    ' "$*" >&2; log "FAILED: $*"; }
blank() { printf '\n'; }
did()  { [ "$DRY_RUN" = 1 ] || good "$@"; }          # a success line for something that really happened (never in a dry run)
# disp TEXT : a command as the user would type it (nightage instead of its full path, ~ for the home folder)
disp() { printf '%s' "$*" | sed "s|^pkg_retry |pkg install -y |; s|^pip_light\$|pip install $LIGHT_PKGS|; s|$NBIN|nightage|g; s|$VENV|~/.nightage-venv|g; s|$HOME|~|g"; }
short_path() { case "$1" in "$HOME"/*) printf '~%s' "${1#"$HOME"}" ;; *) printf '%s' "$1" ;; esac; }

# ---------------------------------------------------------------- percent bar and steps
TOTAL=8
STEP=0
DONE_W=0
weight_of() { # share of the whole install, in percent; sums to 100
  case "$1" in 1) echo 4 ;; 2) echo 14 ;; 3) echo 20 ;; 4) echo 4 ;; 5) echo 10 ;; 6) echo 30 ;; 7) echo 8 ;; *) echo 10 ;; esac
}
make_bar() { # make_bar PCT WIDTH -> BAR (filled in the accent colour, the rest dim; same rounding as the Nightage UI bar)
  mb_n=$(( ($1 * $2 + 50) / 100 )); mb_f=''; mb_e=''; mb_i=0
  while [ "$mb_i" -lt "$2" ]; do
    if [ "$mb_i" -lt "$mb_n" ]; then mb_f="$mb_f$FILL"; else mb_e="$mb_e$EMPTY"; fi
    mb_i=$((mb_i + 1))
  done
  BAR="${ACC}${mb_f}${RST}${DIM}${mb_e}${RST}"
}
# One line:  [████████░░░░░░░░] 50%  Step 3 of 8   (bar width from the terminal width, at least 10 where it fits)
bar_line() { # bar_line PCT LABEL_LONG LABEL_SHORT
  refresh_width
  bl_lim=$((WIDTH - 1)); bl_pct="$1%"; bl_tail=$2
  bl_bw=$((bl_lim - 3 - ${#bl_pct} - 2 - ${#bl_tail}))
  if [ "$bl_bw" -lt 10 ]; then bl_tail=$3; bl_bw=$((bl_lim - 3 - ${#bl_pct} - 2 - ${#bl_tail})); fi
  [ "$bl_bw" -lt 4 ] && bl_bw=4
  [ "$bl_bw" -gt 30 ] && bl_bw=30
  make_bar "$1" "$bl_bw"
  printf '[%s] %s  %s\n' "$BAR" "$bl_pct" "$bl_tail"
}
step() { # step "Title"
  STEP=$((STEP + 1))
  blank
  bar_line "$DONE_W" "Step $STEP of $TOTAL" "$STEP/$TOTAL"
  wrap "${BLD}" '' "$1" | sed "s/\$/${RST}/"
  log "STEP $STEP: $1"
  DONE_W=$((DONE_W + $(weight_of "$STEP")))
}

# ---------------------------------------------------------------- questions
TTY_OK=0
if [ -t 0 ] || { [ -r /dev/tty ] && (: </dev/tty) 2>/dev/null; }; then TTY_OK=1; fi
read_answer() {
  if [ -t 0 ]; then read -r ra_r || ra_r=''; else read -r ra_r </dev/tty || ra_r=''; fi
  printf '%s' "$ra_r"
}
# ask "question" y|n  -> 0 = yes. --yes and --dry-run take the default and say so.
ask() {
  a_d=$2
  if [ "$a_d" = y ]; then a_hint='[Y/n]'; a_word=yes; else a_hint='[y/N]'; a_word=no; fi
  refresh_width
  if [ "$ASSUME_YES" = 1 ] || [ "$DRY_RUN" = 1 ]; then
    if [ "$DRY_RUN" = 1 ] && [ "$ASSUME_YES" != 1 ]; then a_n="dry run, would ask: $a_word"; else a_n="auto: $a_word"; fi
    wrap "  ${ACC}?${RST} " '    ' "$1 ($a_n)"
    log "question: $1 -> $a_word ($a_n)"
    [ "$a_d" = y ]; return
  fi
  [ "$TTY_OK" = 1 ] || die "There is no keyboard to ask you on." "Run again with --yes to accept the defaults, or download the installer and run it: sh install-termux.sh"
  printf '%s ' "$(wrap "  ${ACC}?${RST} " '    ' "$1 $a_hint")"
  a_r=$(read_answer)
  log "question: $1 -> '$a_r'"
  case "$a_r" in [yY]*) return 0 ;; [nN]*) return 1 ;; *) [ "$a_d" = y ] ;; esac
}
# ask_text "question" -> the answer in ANSWER (empty in --yes / --dry-run)
ask_text() {
  ANSWER=''
  if [ "$ASSUME_YES" = 1 ] || [ "$DRY_RUN" = 1 ] || [ "$TTY_OK" != 1 ]; then return 0; fi
  refresh_width
  printf '%s ' "$(wrap "  ${ACC}?${RST} " '    ' "$1")"
  ANSWER=$(read_answer)
  log "question: $1 -> '$ANSWER'"
}

# ---------------------------------------------------------------- running things quietly
SPIN_PID=''
CHILD=''
cleanup() { [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null || true; CHILD=''; }
on_interrupt() {
  cleanup
  [ "$LIVE" = 1 ] && printf '\r%s[K' "$ESC"
  blank
  say "Stopped. Nothing is broken: run the installer again and it continues where it stopped (downloads resume)."
  exit 130
}
trap on_interrupt INT TERM
trap cleanup EXIT

die() { # die "what happened" ["what to do"]
  refresh_width
  bad "$1"
  [ -n "${2:-}" ] && say "$2"
  if [ "$DRY_RUN" != 1 ]; then say "Log: $(short_path "$LOG")"; fi
  exit 1
}

show_tail() { # the last 8 lines the failed command printed, cut to the screen width
  refresh_width
  tail -n 8 "$LAST" 2>/dev/null | while IFS= read -r st_l; do wrap '    ' '    ' "$st_l"; done
}

spin() { # spin PID LABEL : one line, redrawn; static text without a terminal
  refresh_width
  if [ "$LIVE" != 1 ]; then wrap "  ${BUL} " '    ' "$2 ..."; return 0; fi
  sp_room=$((WIDTH - 1 - 4 - 7))                  # "  X " + " mm:ss"
  sp_label=$(printf '%s' "$2" | awk -v n="$sp_room" '{ print substr($0, 1, n) }')
  sp_t0=$(date +%s); sp_i=0
  while kill -0 "$1" 2>/dev/null; do
    if [ "$UNI" = 1 ]; then
      case $((sp_i % 10)) in 0) sp_f='⠋' ;; 1) sp_f='⠙' ;; 2) sp_f='⠹' ;; 3) sp_f='⠸' ;; 4) sp_f='⠼' ;; 5) sp_f='⠴' ;; 6) sp_f='⠦' ;; 7) sp_f='⠧' ;; 8) sp_f='⠇' ;; *) sp_f='⠏' ;; esac
    else
      case $((sp_i % 4)) in 0) sp_f='|' ;; 1) sp_f='/' ;; 2) sp_f='-' ;; *) sp_f='\' ;; esac
    fi
    sp_e=$(( $(date +%s) - sp_t0 ))
    printf '\r%s[K  %s%s%s %s %s%d:%02d%s' "$ESC" "$ACC" "$sp_f" "$RST" "$sp_label" "$DIM" $((sp_e / 60)) $((sp_e % 60)) "$RST"
    sleep 0.2 2>/dev/null || sleep 1
    sp_i=$((sp_i + 1))
  done
  printf '\r%s[K' "$ESC"
}

# run_quiet "Label" CMD ARGS... : output goes to the log, one spinner line is shown; on failure the last 8 log lines and the log path.
# In --dry-run only the command is printed. Returns the command's status (never stops the script by itself).
run_quiet() {
  rq_label=$1; shift
  log "run: $*"
  if [ "$DRY_RUN" = 1 ]; then say "[dry-run] $(disp "$*")"; return 0; fi
  if [ "$VERBOSE" = 1 ]; then
    say "$rq_label"; say "\$ $(disp "$*")"
    if "$@"; then return 0; else return $?; fi
  fi
  "$@" >"$LAST" 2>&1 </dev/null &
  CHILD=$!
  spin "$CHILD" "$rq_label"
  if wait "$CHILD"; then rq_rc=0; else rq_rc=$?; fi
  CHILD=''
  cat "$LAST" >>"$LOG" 2>/dev/null || true
  log "exit code: $rq_rc"
  return "$rq_rc"
}
# must "Label" "hint" CMD... : run_quiet, and stop the install with the log tail when it fails
must() {
  m_label=$1; m_hint=$2; shift 2
  if run_quiet "$m_label" "$@"; then return 0; fi
  refresh_width
  bad "$m_label did not work."
  [ "$VERBOSE" = 1 ] || show_tail
  say "$m_hint"
  say "Full log: $(short_path "$LOG")"
  exit 1
}

# pkg_retry ARGS... : `pkg install -y ARGS`, up to 3 tries (mirrors in Termux sometimes fail once and work a moment later).
# Runs inside run_quiet, so its output goes to the log. UNVERIFIED on a real phone.
pkg_retry() {
  pr_i=1
  while [ "$pr_i" -le 3 ]; do
    if pkg install -y "$@"; then return 0; fi
    echo "pkg install failed (try $pr_i of 3)"
    pr_i=$((pr_i + 1))
    [ "$pr_i" -le 3 ] && { pkg update -y >/dev/null 2>&1 || true; sleep "${NIGHTAGE_RETRY_WAIT:-3}"; }
  done
  return 1
}
LIGHT_PKGS='typer>=0.15 rich>=13.9 pyyaml>=6.0 httpx>=0.28 prompt_toolkit>=3.0.43'
# pip_light : install the light packages. On a very new Python (3.14) a package may have no ready-made build and would need a
# compiler: then try ready-made builds only, then Termux's own python-yaml with system packages visible. UNVERIFIED.
pip_light() {
  # shellcheck disable=SC2086  # the list is meant to be split
  "$VENV/bin/python" -m pip install --disable-pip-version-check --progress-bar off $LIGHT_PKGS && return 0
  echo "retrying with ready-made builds only"
  # shellcheck disable=SC2086
  "$VENV/bin/python" -m pip install --disable-pip-version-check --progress-bar off --only-binary :all: $LIGHT_PKGS && return 0
  echo "retrying with Termux's python-yaml and system packages"
  pkg_retry python-yaml || return 1
  python -m venv --system-site-packages "$VENV" || return 1
  # shellcheck disable=SC2086
  "$VENV/bin/python" -m pip install --disable-pip-version-check --progress-bar off $LIGHT_PKGS
}

# ---------------------------------------------------------------- facts about the phone
ram_mb() {
  if [ -n "${NIGHTAGE_TEST_RAM_MB:-}" ]; then echo "$NIGHTAGE_TEST_RAM_MB"; return; fi
  awk '/^MemTotal:/ { printf "%d\n", $2 / 1024; exit }' /proc/meminfo 2>/dev/null || echo 0
}
free_mb() { df -Pk "$HOME" 2>/dev/null | awk 'NR == 2 { printf "%d\n", $4 / 1024 }'; }
gb() { LC_ALL=C awk -v m="$1" 'BEGIN { printf "%.1f GB", m / 1024 }'; }

# Chat model by memory: the same tiers as agent/profiles.py model_tier (tests/test_install_termux.py keeps them in step).
chat_tier() { # chat_tier RAM_MB -> tag
  if [ "$1" -le 0 ]; then echo qwen3.5:2b
  elif [ "$1" -lt 3072 ]; then echo qwen3.5:0.8b
  elif [ "$1" -lt 6144 ]; then echo qwen3.5:2b
  else echo qwen3.5:4b; fi
}
chat_size() { # approximate download in MB (ollama.com tag pages; UNVERIFIED for your Ollama build)
  case "$1" in qwen3.5:0.8b) echo 1000 ;; qwen3.5:2b) echo 2700 ;; qwen3.5:4b) echo 3400 ;; qwen3.5:9b) echo 6600 ;; *) echo 0 ;; esac
}

# ---------------------------------------------------------------- start
refresh_width
if [ "$DRY_RUN" != 1 ]; then mkdir -p "$STATE_DIR" 2>/dev/null || { LOG=/dev/null; LAST=/dev/null; }; fi
log "---- install-termux.sh $(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null) args: $*"
blank
wrap "${BLD}" '' "Nightage for your phone" | sed "s/\$/${RST}/"
note "Experimental: not tried on a real phone yet. If something fails, the log says why."
note "Quiet by default. Details go to the log (--verbose shows them):"
say "  $(short_path "$LOG")"
[ "$DRY_RUN" = 1 ] && note "Dry run: nothing is changed."

# ---- 1 ------------------------------------------------------------------------------------------------------------------------
step "Checking this phone"
if ! have pkg; then
  [ "$DRY_RUN" = 1 ] || die "This installer is for Termux (no pkg command found)." "On a normal Linux computer use install.sh instead."
  warn "No pkg command here: the dry run goes on anyway."
fi
case "$PREFIX_DIR" in *com.termux*) ;; *) warn "PREFIX does not look like Termux. Going on, but this is untested." ;; esac
[ -f "$SRC/pyproject.toml" ] || die "Run this from a Nightage folder (pyproject.toml not found)."
ARCH=$(uname -m 2>/dev/null || echo unknown)
RAM=$(ram_mb); case $RAM in ''|*[!0-9]*) RAM=0 ;; esac
FREE=$(free_mb); case $FREE in ''|*[!0-9]*) FREE=0 ;; esac
good "Termux on $ARCH"
if [ "$RAM" -gt 0 ]; then good "$(gb "$RAM") memory"; else warn "Could not read the memory size."; fi
if [ "$FREE" -gt 0 ]; then
  if [ "$FREE" -lt 3072 ]; then warn "Only $(gb "$FREE") free: models need 1 to 6 GB."; else good "$(gb "$FREE") free space"; fi
fi
if have python; then good "$(python --version 2>&1)"; fi

# ---- 2 ------------------------------------------------------------------------------------------------------------------------
step "Installing basic tools"
missing=''
for p in python git rg; do have "$p" || missing="$missing $p"; done
if [ -n "$missing" ]; then
  pk=''
  for p in $missing; do case "$p" in rg) pk="$pk ripgrep" ;; *) pk="$pk $p" ;; esac; done
  # shellcheck disable=SC2086  # the package list is meant to be split
  must "Installing$pk" "Try:  pkg update   (or termux-change-repo to pick another mirror), then run this again." pkg_retry $pk
  did "Installed$pk"
else
  good "python, git, ripgrep are there"
fi
have ruff || note "Optional: pkg install ruff gives the code agent its lint check."

# ---- 3 ------------------------------------------------------------------------------------------------------------------------
step "Python environment"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  good "Environment exists ($(short_path "$VENV"))"
else
  if [ -d "$VENV" ]; then note "Found a broken environment: making it again."; run_quiet "Removing the broken environment" rm -rf "$VENV" || true; fi
  must "Creating the environment" "Termux's python package includes venv. Try:  pkg install python" python -m venv "$VENV"
  did "Environment created"
fi
# Keep in step with [project].dependencies in pyproject.toml, minus the optional trio (sqlite-vec, ruff, coverage).
must "Installing 5 light packages" "On a very new Python a package may have no ready-made build yet; see docs/TERMUX.md (Troubleshooting)." pip_light
did "Packages ready (memory search uses keywords on a phone)"

# ---- 4 ------------------------------------------------------------------------------------------------------------------------
step "Command and settings"
if [ "$DRY_RUN" = 1 ]; then
  say "[dry-run] write the nightage command"
else
  mkdir -p "$BIN_DIR"
  tmp="$BIN_DIR/.nightage.$$"
  {
    printf '#!/bin/sh\n'
    printf '# generated by install-termux.sh (run it again to refresh)\n'
    printf 'PYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}" exec "%s/bin/python" -m agent.main "$@"\n' "$SRC" "$VENV"
  } >"$tmp"
  chmod 755 "$tmp"
  mv "$tmp" "$NBIN"
  good "Command ready: nightage"
fi

# cfg_set FORCE KEY VALUE : set a dotted key in config.yaml; an existing value is kept unless FORCE=1. Sets CFG_NOTE.
cfg_set() {
  CFG_NOTE=''
  if [ "$DRY_RUN" = 1 ]; then CFG_NOTE="[dry-run] config: $2 = $3"; return 0; fi
  CFG_NOTE=$(NIGHTAGE_CFG="$CFG" CFG_FORCE="$1" CFG_KEY="$2" CFG_VAL="$3" "$VENV/bin/python" - 2>>"$LOG" <<'PY'
import os, pathlib, yaml
path = pathlib.Path(os.environ["NIGHTAGE_CFG"])
data = yaml.safe_load(path.read_text()) if path.exists() else {}
data = data if isinstance(data, dict) else {}
keys = os.environ["CFG_KEY"].split(".")
node = data
for k in keys[:-1]:
    if not isinstance(node.get(k), dict):
        node[k] = {}
    node = node[k]
last = keys[-1]
if node.get(last) and not int(os.environ["CFG_FORCE"]) and node.get(last) != os.environ["CFG_VAL"]:
    print("kept your existing %s (%s)" % (os.environ["CFG_KEY"], node[last]))
else:
    node[last] = os.environ["CFG_VAL"]
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    print("set %s: %s" % (os.environ["CFG_KEY"], os.environ["CFG_VAL"]))
PY
  ) || { CFG_NOTE="could not write the settings file"; return 1; }
}

# assign_chat TAG : every chat role uses TAG. Prefers `nightage models use` (another part of Nightage); else writes config.yaml.
assign_chat() {
  if "$NBIN" models use --help >/dev/null 2>&1 && "$NBIN" models use "$1" --yes >>"$LOG" 2>&1; then
    good "Every chat role uses $1"; return 0
  fi
  for ac_role in chat background coding reasoning unrestricted weekly; do
    cfg_set 1 "models.$ac_role" "$1" || { warn "$CFG_NOTE"; return 0; }
  done
  good "Every chat role uses $1 (written to config.yaml)"
}
assign_embed() {
  if "$NBIN" models embed --help >/dev/null 2>&1 && "$NBIN" models embed "$1" --yes >>"$LOG" 2>&1; then
    good "Memory model set: $1"; return 0
  fi
  cfg_set 1 models.embed "$1" || { warn "$CFG_NOTE"; return 0; }
  good "Memory model set: $1 (written to config.yaml)"
}

PROFILE=${PROFILE_ARG:-phone}
if [ "$PROFILE" = phone ]; then
  if [ -n "$PROFILE_ARG" ] || ask "Use the phone profile? Small models, short context, frees memory when you leave." y; then
    cfg_set 0 profile phone || warn "$CFG_NOTE"
    [ -n "$CFG_NOTE" ] && did "$CFG_NOTE"
    if [ "$DRY_RUN" = 1 ]; then say "$CFG_NOTE"; fi
  else
    note "Skipped. Try it any time:  nightage --profile phone"
  fi
fi

remote=${NIGHTAGE_REMOTE_HOST:-}
if [ -z "$remote" ] && [ "$NO_OLLAMA" = 0 ] && [ -z "$MODEL_OVERRIDE" ]; then
  if ask "Use Ollama on another machine, e.g. your PC over Tailscale? (No = models on this phone.)" n; then
    ask_text "Address of that Ollama (host or host:port):"
    remote=$ANSWER
  fi
fi
if [ -n "$remote" ]; then
  cfg_set 0 ollama.host "$remote" || warn "$CFG_NOTE"
  did "$CFG_NOTE"
  if [ "$DRY_RUN" = 1 ]; then say "$CFG_NOTE"; fi
  note "On the PC: Ollama must listen on the network (OLLAMA_HOST=0.0.0.0) and Tailscale must run on both."
  NO_OLLAMA=1
fi

# ---- 5 ------------------------------------------------------------------------------------------------------------------------
step "Ollama on this phone"
OLLAMA_UP=0
if [ "$NO_OLLAMA" = 1 ]; then
  note "Skipped: models run on another machine, or you chose --no-ollama."
elif [ "$ARCH" != aarch64 ] && [ "$ARCH" != x86_64 ] && [ "$DRY_RUN" != 1 ]; then
  warn "Termux has no Ollama package for $ARCH (only aarch64 and x86_64). Use Ollama on another machine: set ollama.host."
  NO_OLLAMA=1
else
  if [ "$RAM" -ge 6144 ] || [ "$RAM" -le 0 ]; then od=y; else od=n; fi
  if [ "$RAM" -gt 0 ] && [ "$RAM" -lt 6144 ]; then note "Under 6 GB of memory: on-phone models will be small and slow."; fi
  if ask "Run the AI models on this phone? Needs about 2 to 5 GB of space." "$od"; then
    if have ollama; then
      good "Ollama is installed"
    else
      must "Installing Ollama" "Try:  pkg update   then run this again. Or use another machine: set ollama.host." pkg_retry ollama
      did "Ollama installed"
    fi
    if run_quiet "Starting Ollama" "$NBIN" ollama start; then
      OLLAMA_UP=1
      did "Ollama is running (it stops when you leave Nightage)"
    else
      refresh_width
      warn "Ollama did not start. Nightage retries by itself when you open a chat."
      [ "$VERBOSE" = 1 ] || show_tail
    fi
    if have termux-wake-lock; then
      if ask "Keep the phone awake while Nightage works (termux-wake-lock)?" y; then
        run_quiet "Taking a wake lock" termux-wake-lock || true
        did "Wake lock on (release: termux-wake-unlock)"
      fi
    else
      note "Tip: pkg install termux-api (and the Termux:API app) adds termux-wake-lock, so Android keeps Termux running with the screen off."
    fi
    note "Android may stop background programs. If Ollama keeps dying, see docs/TERMUX.md."
  else
    NO_OLLAMA=1
    note "Skipped. Later:  pkg install ollama   then   nightage pull qwen3.5:4b"
  fi
fi

# ---- 6 ------------------------------------------------------------------------------------------------------------------------
PENDING=''
step "Chat model"
CHAT=''
if [ "$NO_OLLAMA" = 1 ]; then
  note "Skipped: no model on this phone."
else
  CHAT=${MODEL_OVERRIDE:-$(chat_tier "$RAM")}
  csize=$(chat_size "$CHAT")
  good "Chosen for $(gb "$RAM") memory: $CHAT"
  [ "$csize" -gt 0 ] && note "Download about $(gb "$csize"). Runs on the CPU; replies take a while."
  if [ -z "$MODEL_OVERRIDE" ] && [ "$RAM" -ge 10240 ]; then
    note "With this much memory qwen3.5:9b (about 6.6 GB) is possible, but slow. Try it later:  nightage pull qwen3.5:9b"
  fi
  if ask "Download $CHAT now?" y; then
    if [ "$DRY_RUN" = 1 ]; then say "[dry-run] nightage pull $CHAT"
    elif "$NBIN" pull "$CHAT"; then
      good "$CHAT is ready"
      assign_chat "$CHAT"
    else
      warn "The download did not finish."
      PENDING="$PENDING;nightage pull $CHAT"
    fi
  else
    PENDING="$PENDING;nightage pull $CHAT"
    note "Skipped. Later:  nightage pull $CHAT"
  fi
fi

# ---- 7 ------------------------------------------------------------------------------------------------------------------------
step "Memory model"
EMB=''
if [ "$NO_OLLAMA" = 1 ]; then
  note "Skipped: no model on this phone."
elif [ "$NO_EMBED" = 1 ]; then
  note "Skipped (--no-embed): Nightage recalls by keywords."
else
  if [ -n "$EMBED_OVERRIDE" ]; then
    EMB=$EMBED_OVERRIDE
  else
    note "The memory model lets Nightage recall things by meaning. It is small."
    if ask "Do you mostly chat in English?" y; then EMB=all-minilm; else EMB="embeddinggemma:300m-qat-q4_0"; fi
  fi
  case "$EMB" in
    all-minilm) esz=46 ;; "embeddinggemma:300m-qat-q4_0") esz=239 ;; nomic-embed-text) esz=274 ;; *) esz=0 ;;
  esac
  good "Memory model: $EMB"
  [ "$esz" -gt 0 ] && note "Download $esz MB."
  if [ "$DRY_RUN" = 1 ]; then say "[dry-run] nightage pull $EMB"
  elif "$NBIN" pull "$EMB"; then
    good "$EMB is ready"
    assign_embed "$EMB"
  else
    warn "The download did not finish."
    PENDING="$PENDING;nightage embed use $EMB"
  fi
fi

# ---- 8 ------------------------------------------------------------------------------------------------------------------------
step "Health check"
if [ "$DRY_RUN" = 1 ]; then
  say "[dry-run] nightage doctor --only system,ollama,models"
else
  "$NBIN" doctor --only system,ollama,models || note "Some checks need attention; the lines above say what to do."
fi

# ---- the end ------------------------------------------------------------------------------------------------------------------
refresh_width
blank
bar_line 100 "Done" "Done"
blank
box_w=$((WIDTH - 1)); [ "$box_w" -gt 44 ] && box_w=44
inner=$((box_w - 4))
if [ "$UNI" = 1 ]; then TL='╭'; TR='╮'; BL='╰'; BR='╯'; HZ='─'; VT='│'; else TL='+'; TR='+'; BL='+'; BR='+'; HZ='-'; VT='|'; fi
hz=''; i=0; while [ "$i" -lt $((box_w - 2)) ]; do hz="$hz$HZ"; i=$((i + 1)); done
boxline() { # boxline TEXT : wrapped inside the frame, padded on the right
  WIDTH_SAVE=$WIDTH; WIDTH=$((inner + 1))
  wrap '' '' "$1" | awk -v w="$inner" -v vt="$VT" '{ printf "%s %-" w "s %s\n", vt, $0, vt }'
  WIDTH=$WIDTH_SAVE
}
printf '%s%s%s\n' "$TL" "$hz" "$TR"
boxline "Done. Start with: nightage"
if [ "$DRY_RUN" = 1 ]; then boxline "(dry run: nothing was changed)"; fi
if [ "$NO_OLLAMA" = 0 ] && [ -z "$PENDING" ]; then boxline "Models stop when you leave."; fi
printf '%s%s%s\n' "$BL" "$hz" "$BR"
blank
if [ -n "$PENDING" ]; then
  say "Still to do (run these when you have a connection):"
  OLD_IFS=$IFS; IFS=';'
  for todo in $PENDING; do [ -n "$todo" ] && note "$todo"; done
  IFS=$OLD_IFS
fi
say "Help: docs/TERMUX.md. Android sometimes stops background programs; the guide shows what to do."
note "Optional: termux-setup-storage lets Nightage read your Downloads folder."
