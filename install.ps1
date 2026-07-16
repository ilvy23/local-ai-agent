# agent installer — Windows (PowerShell).
#
# Idempotent: re-running is safe, it only does what's missing.
#
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#   powershell -ExecutionPolicy Bypass -File .\install.ps1 -Yes   (no prompts)
#
# EXPERIMENTAL: written carefully but not yet run on a real Windows machine.
# If it breaks, please say so:
# https://github.com/ilvy23/local-ai-agent/issues/1

param([switch]$Yes)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$script:Step  = 0
$script:Total = 4
# Don't block on a prompt when there's no console to type into (CI, pipes).
$script:AutoYes = $Yes -or [Console]::IsInputRedirected

function Step($m) {
    $script:Step++
    Write-Host ""
    Write-Host "[$script:Step/$script:Total] " -ForegroundColor Cyan -NoNewline
    Write-Host $m
}
function Ok($m)   { Write-Host "   " -NoNewline; Write-Host "OK " -ForegroundColor Green -NoNewline; Write-Host $m }
function Info($m) { Write-Host "   |  " -ForegroundColor DarkGray -NoNewline; Write-Host $m }
function Warn($m) { Write-Host "   " -NoNewline; Write-Host "!  " -ForegroundColor Yellow -NoNewline; Write-Host $m }
function Die($m)  { Write-Host "   " -NoNewline; Write-Host "X  " -ForegroundColor Red -NoNewline; Write-Host $m; exit 1 }
function Have($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

function Ask($q) {   # default yes
    if ($script:AutoYes) { Write-Host "   ?  $q [yes]" -ForegroundColor Cyan; return $true }
    Write-Host "   ?  " -ForegroundColor Cyan -NoNewline
    $r = Read-Host "$q [Y/n]"
    return ($r -notmatch '^\s*[nN]')
}

function OllamaUp {
    try { Invoke-RestMethod "http://localhost:11434/api/version" -TimeoutSec 3 | Out-Null; return $true }
    catch { return $false }
}

Write-Host ""
Write-Host " +---------------------------------------------+" -ForegroundColor Cyan
Write-Host " |   agent - a local AI in your terminal       |" -ForegroundColor Cyan
Write-Host " |   runs on your machine. nothing phones home |" -ForegroundColor DarkGray
Write-Host " +---------------------------------------------+" -ForegroundColor Cyan

# -- 1. system ---------------------------------------------------------------
Step "Checking your system"
$os = (Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue).Caption
if ($os) { Ok $os } else { Ok "Windows" }
Ok "PowerShell $($PSVersionTable.PSVersion)"

# -- 2. python toolchain -----------------------------------------------------
Step "Python toolchain"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (Have "uv") {
    Ok "uv $((uv --version) -replace '^uv\s+','') already installed"
} else {
    Info "uv installs Python and this project's dependencies, into your"
    Info "user profile - no system-wide changes."
    Info "source: https://astral.sh/uv/install.ps1"
    if (-not (Ask "Install uv?")) { Die "uv is required. See https://docs.astral.sh/uv/" }
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex" | Out-Null
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Have "uv")) { Die "uv installed but isn't on PATH - open a new PowerShell and re-run." }
    Ok "uv installed"
}

Info "installing dependencies (first run also fetches a matching Python)..."
uv sync --quiet
Ok "dependencies ready"

# -- 3. ollama ---------------------------------------------------------------
Step "Ollama"
if (-not (Have "ollama")) {
    # Installed but not on PATH is common on Windows; look in the usual place.
    $guess = "$env:LOCALAPPDATA\Programs\Ollama"
    if (Test-Path "$guess\ollama.exe") { $env:Path = "$guess;$env:Path" }
}

if (Have "ollama") {
    Ok "Ollama already installed"
} else {
    Warn "Ollama is not installed"
    Info "It's what actually runs the AI models, locally on this machine -"
    Info "the reason none of your conversations leave your computer."
    Info "official download: https://ollama.com/download/OllamaSetup.exe"
    if (Ask "Download and install Ollama now? (~700 MB installer)") {
        if (Have "winget") {
            Info "installing via winget..."
            winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        } else {
            $exe = Join-Path $env:TEMP "OllamaSetup.exe"
            Info "downloading OllamaSetup.exe..."
            Invoke-WebRequest "https://ollama.com/download/OllamaSetup.exe" -OutFile $exe
            Info "running the installer (follow its prompts)..."
            Start-Process -FilePath $exe -Wait
            Remove-Item $exe -ErrorAction SilentlyContinue
        }
        $env:Path = "$env:LOCALAPPDATA\Programs\Ollama;$env:Path"
        if (-not (Have "ollama")) {
            Warn "Ollama installed, but this shell can't see it yet."
            Info "Open a NEW PowerShell window and re-run this script."
            exit 0
        }
        Ok "Ollama installed"
    } else {
        Write-Host ""
        Info "No problem - grab it from https://ollama.com/download"
        Info "then re-run this script. It picks up where it left off."
        exit 0
    }
}

if (OllamaUp) {
    Ok "server is running"
} else {
    Info "starting the Ollama server..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    foreach ($i in 1..15) { if (OllamaUp) { break }; Start-Sleep -Seconds 1 }
    if (OllamaUp) { Ok "server is running" }
    else { Warn "couldn't start it - run 'ollama serve' in another window" }
}

# -- 4. models ---------------------------------------------------------------
Step "Models"
$missing = @()
foreach ($m in @("qwen2.5:7b", "bge-m3")) {
    $short = $m.Split(":")[0]
    if ((ollama list 2>$null) -match [regex]::Escape($short)) {
        Ok "$m already downloaded"
    } else {
        $missing += $m
    }
}

if ($missing.Count -gt 0) {
    Info "qwen2.5:7b (4.7 GB) does the thinking - it must be a model that"
    Info "supports tools, or the agent can't read files or search."
    Info "bge-m3 (1.2 GB) powers the memory."
    Info "missing: $($missing -join ' ')"
    if (Ask "Download them now? (one time)") {
        foreach ($m in $missing) {
            Info "pulling $m..."
            ollama pull $m
            Ok "$m ready"
        }
    } else {
        Warn "skipped - agent won't work until you run:"
        foreach ($m in $missing) { Info "  ollama pull $m" }
    }
}

# -- done --------------------------------------------------------------------
Write-Host ""
Write-Host " +---------------------------------------------+" -ForegroundColor Green
Write-Host " |   All set.                                  |" -ForegroundColor Green
Write-Host " |                                             |" -ForegroundColor Green
Write-Host " |     uv run agent menu   <- start here       |" -ForegroundColor Green
Write-Host " |     uv run agent chat   <- straight to chat |" -ForegroundColor Green
Write-Host " +---------------------------------------------+" -ForegroundColor Green
Write-Host ""
Write-Host "   Tip: end any message with /web to search the internet." -ForegroundColor DarkGray
Write-Host ""
