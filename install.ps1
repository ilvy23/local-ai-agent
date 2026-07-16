# agent installer — Windows (PowerShell).
#
# Idempotent. Installs uv (which fetches the right Python), syncs dependencies,
# ensures Ollama is installed and running, and pulls the default models.
#
# Run from an ordinary PowerShell prompt:
#     powershell -ExecutionPolicy Bypass -File .\install.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($m)  { Write-Host "▸ $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "! $m" -ForegroundColor Yellow }

# --- uv: manages the Python toolchain + the virtualenv ---------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "installing uv…"
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH — open a new PowerShell window and re-run this script."
}

# --- dependencies (uv downloads the pinned Python automatically) -----------
Say "installing dependencies (first run also fetches Python)…"
uv sync

# --- Ollama: the local model server ----------------------------------------
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Say "installing Ollama…"
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
    } else {
        Warn "winget not found. Install Ollama from https://ollama.com/download, then re-run."
        exit 1
    }
}

# start the server if it isn't answering
try {
    Invoke-RestMethod "http://localhost:11434/api/version" -TimeoutSec 2 | Out-Null
} catch {
    Say "starting Ollama in the background…"
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# --- models (skip any already present) -------------------------------------
foreach ($m in @("qwen2.5:7b", "bge-m3")) {
    $short = $m.Split(":")[0]
    if ((ollama list 2>$null) -match $short) {
        Say "model $m already present"
    } else {
        Say "pulling $m (a few GB, one time)…"
        ollama pull $m
    }
}

Write-Host ""
Say "done!  start agent with:"
Write-Host "    uv run agent menu     # interactive menu"
Write-Host "    uv run agent chat     # jump into a chat"
