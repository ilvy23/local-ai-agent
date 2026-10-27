<#
.SYNOPSIS
  Nightage guided installer for Windows (PowerShell 5.1 and 7). UNVERIFIED on a real Windows machine: written carefully,
  syntax-checked and statically tested on Linux only. If it breaks, please send the log file it prints.

.DESCRIPTION
  One-liner (open "Windows PowerShell" from the Start menu, paste, press Enter, then just answer the questions;
  every question has a safe default):
      irm <INSTALL_URL>/install.ps1 | iex
  With options (the one-liner above cannot take options):
      & ([scriptblock]::Create((irm <INSTALL_URL>/install.ps1))) -Yes
  Careful way (download, read, then run) - recommended if you like to see what runs:
      irm <INSTALL_URL>/install.ps1 -OutFile install.ps1
      notepad install.ps1
      powershell -ExecutionPolicy Bypass -File install.ps1        (works even when scripts are blocked by policy)

  What it does, in plain words (it tells you again, step by step, and asks before each change):
    1. checks your system (Windows version, memory, graphics card)
    2. installs uv (a small official tool from astral.sh) so YOU never install Python by hand
    3. installs Nightage as an isolated tool: its own private Python, nothing mixed into your system
    4. helps you install Ollama (the program that runs the AI model on your computer)
    5. picks ONE model that fits this computer and offers to download it
    6. runs "nightage doctor" and tells you what to type next
  Everything is per-user. No administrator rights are needed (and it refuses to run as Administrator unless -Force).
  Safe to run again (repair / upgrade).

.PARAMETER Yes        Accept the safe default for every question (never deletes your data).
.PARAMETER DryRun     Show everything that would happen, change nothing.
.PARAMETER NoOllama   Do not touch Ollama (use this if it runs on another machine).
.PARAMETER NoModel    Do not offer a model download.
.PARAMETER Model      Use this Ollama model instead of the recommended one.
.PARAMETER Embed      Use this memory (embedding) model, e.g. all-minilm; without it you choose from a short list.
.PARAMETER NoEmbed    Do not offer a memory model (Nightage then recalls by keywords only).
.PARAMETER WithOllama Say yes to installing Ollama.
.PARAMETER Source     Install from this folder, https:// archive URL (.tar.gz / .zip) or git+https URL.
.PARAMETER Uninstall  Remove Nightage (asks before touching your data).
.PARAMETER Force      Allow running as Administrator (not recommended).
.PARAMETER Help       Show this help.
Environment: NIGHTAGE_REPO (project page URL), NIGHTAGE_PYTHON (default 3.12), NIGHTAGE_INSTALL_LOG, NO_COLOR.
Log for bug reports: %LOCALAPPDATA%\nightage\install.log (the path is printed).
#>
param(
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$NoOllama,
    [switch]$NoModel,
    [switch]$NoEmbed,
    [switch]$WithOllama,
    [switch]$Force,
    [switch]$Uninstall,
    [switch]$Help,
    [string]$Source = "",
    [string]$Model = "",
    [string]$Embed = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$global:LASTEXITCODE = 0

# Project page; must equal agent/brand.py REPO_URL (a test checks that). While it is still the bare placeholder the installer
# needs -Source, NIGHTAGE_REPO, or to be run from a downloaded copy of the project.
$RepoUrl      = "https://github.com/ilvy23/local-ai-agent"
$UvUrl        = "https://astral.sh/uv/install.ps1"
$OllamaExeUrl = "https://ollama.com/download/OllamaSetup.exe"
$OllamaDlPage = "https://ollama.com/download"
$ToolName     = "agent"    # the Python package name; it provides the commands nightage, lantern and agent

$script:Step  = 0
$script:Total = 7
$script:Tmp   = $null
$script:Transcribing = $false
$LocalAppData = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $HOME "AppData\Local" }
$RoamingData  = if ($env:APPDATA) { $env:APPDATA } else { Join-Path $HOME "AppData\Roaming" }
$LogFile = if ($env:NIGHTAGE_INSTALL_LOG) { $env:NIGHTAGE_INSTALL_LOG } else { Join-Path (Join-Path $LocalAppData "nightage") "install.log" }

# ---------------------------------------------------------------- help
if ($Help) { Get-Help $PSCommandPath -Detailed; return }   # needs the script saved as a file; see the options in the header

# ---------------------------------------------------------------- looks
$UseColor = ($null -eq $env:NO_COLOR) -or ($env:NO_COLOR -eq "")
function Say([string]$Text, [string]$Color = "") {
    if ($UseColor -and $Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
}
function Show-Step([string]$Text) {
    $script:Step++
    Write-Host ""
    Say ("[{0}/{1}] {2}" -f $script:Step, $script:Total, $Text) "Cyan"
}
function Ok([string]$Text)   { Say ("   + " + $Text) "Green" }
function Info([string]$Text) { Say ("   | " + $Text) "DarkGray" }
function Warn([string]$Text) { Say ("   ! " + $Text) "Yellow" }
function Stop-Install([string]$Text) {
    Say ("   x " + $Text) "Red"
    if (-not $DryRun) { Say ("   Log: " + $LogFile) "Red" }
    Remove-Temp
    Stop-Log
    exit 1
}
function Have([string]$Cmd) { [bool](Get-Command $Cmd -ErrorAction SilentlyContinue) }

function Remove-Temp {
    # Only ever our own freshly created temporary folder.
    if ($script:Tmp -and (Test-Path -LiteralPath $script:Tmp)) {
        Remove-Item -LiteralPath $script:Tmp -Recurse -ErrorAction SilentlyContinue
    }
    $script:Tmp = $null
}
function Stop-Log {
    if ($script:Transcribing) { try { Stop-Transcript | Out-Null } catch { } ; $script:Transcribing = $false }
}

# ---------------------------------------------------------------- questions
# Ask-User "question" "y"|"n"  -> $true / $false. One question at a time; the default is what Enter (or -Yes) picks.
function Ask-User([string]$Question, [string]$Default = "y") {
    $hint = if ($Default -eq "y") { "[Y/n]" } else { "[y/N]" }
    if ($Yes -or $DryRun) {
        $answer = if ($Default -eq "y") { "yes" } else { "no" }
        $note = if ($DryRun -and -not $Yes) { "dry run: would ask, default" } else { "auto" }
        Say ("   ? {0} [{1}: {2}]" -f $Question, $note, $answer) "Cyan"
        return ($Default -eq "y")
    }
    $interactive = $true
    try { if ([Console]::IsInputRedirected) { $interactive = $false } } catch { }
    if (-not $interactive) {
        Stop-Install "There is no terminal to ask you on. Run again with -Yes to accept the defaults, or download the script and run it in a PowerShell window."
    }
    $reply = Read-Host ("   ? {0} {1}" -f $Question, $hint)
    if ($null -eq $reply) { $reply = "" }
    if ($reply -match '^\s*[yY]') { return $true }
    if ($reply -match '^\s*[nN]') { return $false }
    return ($Default -eq "y")
}

# Invoke-Shown exe args... : prints the exact command first; in a dry run it is only shown. Returns the exit code.
function Invoke-Shown([string]$Exe, [string[]]$Arguments = @()) {
    Say ("   $ " + $Exe + " " + ($Arguments -join " ")) "DarkGray"
    if ($DryRun) { Say "     (dry run: not executed)" "DarkGray"; return 0 }
    $global:LASTEXITCODE = 0
    & $Exe @Arguments | Out-Host
    return $global:LASTEXITCODE
}
# Probe a program quietly; never throws. Returns @{ Code = n; Out = "text" }.
function Invoke-Quiet([string]$Exe, [string[]]$Arguments = @()) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $global:LASTEXITCODE = 0
        $out = & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
        return @{ Code = $global:LASTEXITCODE; Out = (($out | Out-String).Trim()) }
    } catch {
        return @{ Code = 1; Out = "" }
    } finally {
        $ErrorActionPreference = $old
    }
}

function Get-FileSha256([string]$Path) {
    try { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower() } catch { return "(could not compute)" }
}
function Format-Size([double]$Mb) {
    if ($Mb -ge 1024) { return ("{0:N1} GB" -f ($Mb / 1024)) }
    return ("{0:N0} MB" -f $Mb)
}
function Get-Download([string]$Url, [string]$OutFile) {
    if ($Url -notmatch '^https://') { Stop-Install ("Refusing to download from a non-HTTPS address: " + $Url) }
    Say ("   $ download " + $Url) "DarkGray"
    if ($DryRun) { Say "     (dry run: not executed)" "DarkGray"; return }
    try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }
    $old = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'      # the progress bar makes Windows PowerShell 5.1 downloads very slow
    try { Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing }
    finally { $ProgressPreference = $old }
}
function Update-SessionPath {
    # Pick up PATH changes made by installers during this session (user + machine), keep what we have.
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $extra = (Join-Path $HOME ".local\bin") + ";" + (Join-Path $LocalAppData "Programs\Ollama")
    $env:Path = (@($user, $machine, $extra, $env:Path) | Where-Object { $_ }) -join ";"
}

function Test-PlaceholderUrl([string]$Url) {
    return -not ($Url -match '^https://github\.com/[^/]+/[^/]+')
}

# ---------------------------------------------------------------- safety: not as Administrator
$isAdmin = $false
try {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $isAdmin = ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
} catch { }
if ($isAdmin -and -not $Force) {
    Say "x This installer is meant for a normal user account, not an Administrator window." "Red"
    Say "  Everything it does lives in your own user profile. Close this window, open a normal" "Red"
    Say "  PowerShell (Start menu, no 'Run as administrator') and run it again, or add -Force." "Red"
    exit 1
}

if (-not $DryRun) {
    try {
        $logDir = Split-Path -Parent $LogFile
        if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
        Start-Transcript -Path $LogFile -Append | Out-Null
        $script:Transcribing = $true
    } catch { }
    try { $script:Tmp = Join-Path ([IO.Path]::GetTempPath()) ("nightage-install-" + [Guid]::NewGuid().ToString("N")); New-Item -ItemType Directory -Path $script:Tmp | Out-Null }
    catch { Stop-Install "Could not make a temporary folder." }
}

# ---------------------------------------------------------------- system facts
function Get-RamMb {
    if ($env:NIGHTAGE_TEST_RAM_MB) { return [int]$env:NIGHTAGE_TEST_RAM_MB }    # test hook: pretend this much RAM
    try { return [int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB) } catch { return 0 }
}
function Get-GpuInfo {
    $r = @{ Desc = "no dedicated graphics card detected (the model will run on the CPU)"; VramMb = 0 }
    if ($env:NIGHTAGE_TEST_VRAM_MB) { $r.VramMb = [int]$env:NIGHTAGE_TEST_VRAM_MB; $r.Desc = "(test) graphics card"; return $r }
    if (Have "nvidia-smi") {
        $q = Invoke-Quiet "nvidia-smi" @("--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
        if ($q.Code -eq 0 -and $q.Out) {
            $total = 0
            foreach ($line in ($q.Out -split "`r?`n")) {
                $parts = $line -split ",\s*"
                if ($parts.Count -ge 2) { $v = 0; if ([int]::TryParse($parts[-1].Trim(), [ref]$v)) { $total += $v } }
            }
            $r.VramMb = $total
            $r.Desc = "NVIDIA: " + (($q.Out -split "`r?`n")[0]) + " MB (all cards: $total MB)"
            return $r
        }
    }
    try {
        $names = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name } | Where-Object { $_ })
        if ($names.Count -gt 0) { $r.Desc = ($names -join "; ") }
    } catch { }
    return $r
}
# Same tiers as `nightage doctor --deep` (agent/doctor/checks/deep.py): GPU memory x0.9, else half of the RAM.
function Select-Model([int]$RamMb, [int]$VramMb) {
    if ($VramMb -gt 0) { $budget = [int]($VramMb * 0.9); $src = "graphics memory" }
    elseif ($RamMb -gt 0) { $budget = [int]($RamMb / 2); $src = "half of the RAM" }
    else { $budget = 4096; $src = "unknown memory (assuming 4 GB)" }
    if     ($budget -lt 3072)  { $m = "qwen3.5:0.8b"; $size = 1300 }
    elseif ($budget -lt 8192)  { $m = "qwen3.5:4b";   $size = 4000 }
    elseif ($budget -lt 20480) { $m = "qwen3.5:9b";   $size = 7600 }
    else                       { $m = "qwen3.5:27b";  $size = 20000 }
    return @{ Model = $m; SizeMb = $size; BudgetMb = $budget; Source = $src }
}
function Test-ModelInstalled([string]$Name) {
    $q = Invoke-Quiet "ollama" @("list")
    if ($q.Code -ne 0) { return $false }
    $names = @($q.Out -split "`r?`n" | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] })
    $want = if ($Name -match ':') { $Name } else { "${Name}:latest" }
    return ($names -contains $want)
}

# ---------------------------------------------------------------- uninstall
function Invoke-Uninstall {
    Say ""
    Say " Nightage uninstaller" "Cyan"
    Info "This removes the Nightage program. It never removes uv, Python or Ollama (how to do that is printed at the end)."
    Update-SessionPath
    if (Have "uv") {
        $list = Invoke-Quiet "uv" @("tool", "list")
        if ($list.Out -match ("(?m)^" + [regex]::Escape($ToolName) + " ")) {
            if (Ask-User ("Remove the Nightage program (uv tool uninstall " + $ToolName + ")?") "y") { [void](Invoke-Shown "uv" @("tool", "uninstall", $ToolName)) }
        } else { Info "No uv-installed Nightage was found." }
    } else { Info "uv was not found, so there is no uv-installed Nightage to remove." }

    $dirs = @()
    $nh = if ($env:NIGHTAGE_HOME) { $env:NIGHTAGE_HOME } else { Join-Path $LocalAppData "nightage" }
    foreach ($d in @($nh, (Join-Path $LocalAppData "agent"), (Join-Path $RoamingData "agent"))) {
        if ((Test-Path -LiteralPath $d) -and ($dirs -notcontains $d)) { $dirs += $d }
    }
    if ($dirs.Count -gt 0) {
        Say ""
        Info "Your Nightage data (settings, memory database, history, plugins) lives here:"
        foreach ($d in $dirs) {
            $bytes = 0
            try { $bytes = (Get-ChildItem -LiteralPath $d -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum } catch { }
            if (-not $bytes) { $bytes = 0 }
            Info ("  {0}  ({1})" -f $d, (Format-Size ($bytes / 1MB)))
        }
        Warn "Deleting these cannot be undone. The default is to KEEP them."
        if (Ask-User "Delete ALL of these folders too?" "n") {
            if ($Yes) { Info "(-Yes never deletes data; skipping)" }
            else {
                $conf = Read-Host "   Type the word delete to confirm"
                if ($conf -eq "delete") {
                    foreach ($d in $dirs) {
                        # Only the folders listed above, never a drive root or the profile itself.
                        if ($d.TrimEnd('\') -eq $HOME.TrimEnd('\') -or $d.Length -le 3) { Warn ("skipping unsafe path: " + $d); continue }
                        Say ("   $ Remove-Item -Recurse " + $d) "DarkGray"
                        if (-not $DryRun) { Remove-Item -LiteralPath $d -Recurse }
                    }
                } else { Info "Not confirmed - nothing deleted." }
            }
        } else { Info "Your data was kept." }
    }
    Say ""
    Info "To also remove uv:     winget uninstall astral-sh.uv   (or delete uv.exe and uvx.exe in $HOME\.local\bin), then: uv python dir  (delete that folder)"
    Info "To remove Ollama:      Settings > Apps > Installed apps > Ollama > Uninstall"
    Info "Downloaded models:     ollama rm <name>   (list them with: ollama list)"
    Ok "Done."
}

# ================================================================ main
Say ""
Say " Nightage installer  (a local AI in your terminal; nothing leaves your computer)" "Cyan"
if ($DryRun) { Say "   DRY RUN: nothing will be changed, everything is only shown." "Yellow" }
if ($Uninstall) { Invoke-Uninstall; Remove-Temp; Stop-Log; return }

# ---- 1. what will happen
Show-Step "What will happen"
Info "This guided installer will, step by step and only after asking you:"
Info "  - check your computer (read only)"
Info "  - install uv (official tool from astral.sh, into your user folder) - it brings its own Python"
Info "  - install Nightage in its own private environment (managed by uv)"
Info "  - help you install Ollama, which runs the AI model on your computer"
Info "  - download ONE AI model that fits your computer (several GB; you see the size first)"
Info "It never needs administrator rights, never edits system settings, and prints every command before running it."
Info "The only setting it changes is YOUR user PATH (so typing 'nightage' works), and only after asking."
Info ("Log file (for bug reports): " + $LogFile)
if (-not (Ask-User "Ready to start?" "y")) { Say "Nothing was changed. Bye."; Remove-Temp; Stop-Log; return }

# ---- 2. detect
Show-Step "Looking at your computer"
$osName = "Windows"
try { $osName = (Get-CimInstance Win32_OperatingSystem).Caption } catch { }
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
Ok ("System: {0}, {1}" -f $osName, $arch)
Ok ("PowerShell " + $PSVersionTable.PSVersion)
$ram = Get-RamMb
if ($ram -gt 0) { Ok ("Memory (RAM): " + (Format-Size $ram)) } else { Warn "Could not read the amount of RAM (assuming a small computer)" }
$gpu = Get-GpuInfo
Ok ("Graphics: " + $gpu.Desc)
$py = Invoke-Quiet "python" @("--version")
if ($py.Code -eq 0 -and $py.Out -match 'Python 3') { Info ("Python already on this computer: " + $py.Out + " - not needed, uv brings its own") }
else { Info "No (working) Python found - that is fine, uv brings its own" }
try {
    $drive = (Get-Item -LiteralPath $HOME).PSDrive
    if ($drive -and $drive.Free) { Info ("Free disk space on " + $drive.Name + ": " + (Format-Size ($drive.Free / 1MB))) }
} catch { }
if ($arch -and $arch -notin @("AMD64", "ARM64", "x86")) { Warn ("Unusual processor type (" + $arch + "): some parts may not have ready-made downloads.") }
if ($env:HTTPS_PROXY -or $env:HTTP_PROXY) { Info "A proxy is set in your environment; downloads will use it if the tools support it (see docs/INSTALL.md > proxy)." }
Update-SessionPath

# ---- 3. uv
Show-Step "Python, the easy way (uv)"
if (Have "uv") {
    $v = Invoke-Quiet "uv" @("--version")
    Ok ("uv is already installed: " + $v.Out)
} else {
    Info "Nightage needs Python 3.11 or newer. You do not have to install it: uv (a small, widely used tool"
    Info "from Astral) downloads and manages one privately for Nightage."
    Info ("It will download this script over HTTPS and run it as you (no admin):  " + $UvUrl)
    Info "It installs uv.exe and uvx.exe into $HOME\.local\bin and adds that folder to YOUR user PATH (no system-wide change)."
    if (Ask-User "Install uv now?" "y") {
        $uvScript = if ($script:Tmp) { Join-Path $script:Tmp "uv-install.ps1" } else { Join-Path ([IO.Path]::GetTempPath()) "uv-install.ps1" }
        Get-Download $UvUrl $uvScript
        if (-not $DryRun) {
            $lines = @(Get-Content -LiteralPath $uvScript).Count
            Info ("Downloaded {0} lines, sha256 {1}" -f $lines, (Get-FileSha256 $uvScript))
            Info "(uv's own installer verifies the program it downloads against checksums published with each release.)"
            if (Ask-User "Read the downloaded script before it runs?" "n") {
                Get-Content -LiteralPath $uvScript | more
                if (-not (Ask-User "Still run it?" "y")) { Stop-Install "Stopped at your request. Nothing else was changed." }
            }
        }
        $psExe = try { (Get-Process -Id $PID).Path } catch { "powershell" }
        $code = Invoke-Shown $psExe @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $uvScript)
        if ($code -ne 0) {
            Warn "The uv installer reported a problem."
            if (Have "winget") {
                Info "Alternative: winget install --id astral-sh.uv -e"
                if (Ask-User "Try winget instead?" "y") { $code = Invoke-Shown "winget" @("install", "--id", "astral-sh.uv", "-e", "--accept-source-agreements", "--accept-package-agreements") }
            }
            if ($code -ne 0) { Stop-Install "Could not install uv. You can install it by hand: https://docs.astral.sh/uv/getting-started/installation/ then run this again." }
        }
        Update-SessionPath
        if (-not $DryRun -and -not (Have "uv")) { Stop-Install "uv was installed but is not found yet. Open a NEW PowerShell window and run this installer again." }
        Ok "uv installed"
    } else {
        Stop-Install "uv is needed (it is how Python gets installed for you). See https://docs.astral.sh/uv/ or run again and say yes."
    }
}

# ---- 4. nightage
Show-Step "Installing Nightage"
$src = ""; $srcDesc = ""
if ($Source) {
    if (Test-Path -LiteralPath $Source) { $src = (Resolve-Path -LiteralPath $Source).Path; $srcDesc = "your -Source: $src" }
    elseif ($Source -match '^(git\+)?https://') { $src = "$ToolName @ $Source"; $srcDesc = "your -Source: $Source" }
    else { Stop-Install ("-Source must be an existing folder/file, an https:// archive URL, or a git+https:// URL (got: " + $Source + ")") }
} else {
    $repo = $env:NIGHTAGE_REPO
    $here = $null
    try { if ($PSCommandPath) { $here = Split-Path -Parent $PSCommandPath } } catch { }
    $pyproject = if ($here) { Join-Path $here "pyproject.toml" } else { $null }
    if (-not $repo -and $pyproject -and (Test-Path -LiteralPath $pyproject) -and (Select-String -LiteralPath $pyproject -Pattern ('^name = "' + $ToolName + '"') -Quiet)) {
        $src = $here; $srcDesc = "this downloaded copy of the project: $here"
    } else {
        if (-not $repo) { $repo = $RepoUrl }
        if (Test-PlaceholderUrl $repo) {
            Warn "The project's download address is not set in this installer yet."
            Info "Tell it where Nightage is:  -Source <folder or https://... archive URL>"
            Info "or set  `$env:NIGHTAGE_REPO = 'https://github.com/<owner>/<repo>'  (the project page; see docs/INSTALL.md)"
            Stop-Install "No place to install Nightage from."
        }
        if ($repo -notmatch '^https://') { Stop-Install ("The project address must be https:// (got: " + $repo + ")") }
        $repo = $repo.TrimEnd('/') -replace '\.git$', ''
        $src = "$ToolName @ $repo/archive/HEAD.tar.gz"
        $srcDesc = "the project page $repo (latest version, as a plain archive: git is not needed)"
    }
}
Info ("Source: " + $srcDesc)
Info "Nightage gets its own private Python environment; it cannot disturb other Python programs."
$pyVer = if ($env:NIGHTAGE_PYTHON) { $env:NIGHTAGE_PYTHON } else { "3.12" }
if (Ask-User "Install Nightage now?" "y") {
    $code = Invoke-Shown "uv" @("tool", "install", "--force", "--python", $pyVer, $src)
    if ($code -ne 0) { Stop-Install ("Installing Nightage failed. The messages above say why; a copy is in " + $LogFile + ". Common causes: no internet, a proxy or antivirus blocking the download (see docs/INSTALL.md), or a wrong -Source.") }
    Ok "Nightage installed"
} else {
    Say "Stopped before installing Nightage. Nothing else was changed."; Remove-Temp; Stop-Log; return
}

# make sure the `nightage` command is on the user PATH
$binDir = Join-Path $HOME ".local\bin"
if (-not $DryRun) {
    $q = Invoke-Quiet "uv" @("tool", "dir", "--bin")
    if ($q.Code -eq 0 -and $q.Out) { $binDir = ($q.Out -split "`r?`n")[0].Trim() }
}
$nightageExe = Join-Path $binDir "nightage.exe"
if (-not $DryRun -and -not (Test-Path -LiteralPath $nightageExe)) { Warn ("Expected the nightage command at " + $nightageExe + " but did not find it.") }
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$onUserPath = $false
if ($userPath) { $onUserPath = (@($userPath -split ";" | ForEach-Object { $_.TrimEnd('\') }) -contains $binDir.TrimEnd('\')) }
if (-not $onUserPath) {
    Warn ($binDir + " is not on your user PATH yet, so typing 'nightage' in a new window would say 'not recognized'.")
    Info "uv can add it for you: it changes only YOUR user PATH setting (Environment > Path for your account), nothing system-wide."
    if (Ask-User "Add it now (uv tool update-shell)?" "y") {
        $code = Invoke-Shown "uv" @("tool", "update-shell")
        if ($code -ne 0) { Warn ("Could not update PATH. Add this folder yourself: Start menu > 'Edit environment variables for your account' > Path > New > " + $binDir) }
        Info "Open a NEW PowerShell window afterwards so the change is picked up."
    } else {
        Info ("No problem. Add " + $binDir + " to your user PATH yourself (Start menu > 'Edit environment variables for your account').")
    }
}
$env:Path = $binDir + ";" + $env:Path

# ---- 5. ollama
Show-Step "Ollama (the engine that runs the AI)"
Update-SessionPath
$ollamaReady = $false
if ($NoOllama) {
    Info "Skipped (-NoOllama). Nightage can use an Ollama on another computer: set ollama.host in its settings."
} elseif (Have "ollama") {
    $ov = Invoke-Quiet "ollama" @("--version")
    Ok ("Ollama is already installed: " + $ov.Out)
    $ollamaReady = $true
} else {
    Warn "Ollama is not installed. It is the program that actually runs the AI model, on your own computer."
    Info "Ollama's Windows installer works for your user only (no administrator rights)."
    $viaWinget = $false
    if (Have "winget") {
        Info "Option A (recommended): winget install --id Ollama.Ollama -e   (Microsoft's package manager, official package)"
        Info ("Option B: download " + $OllamaExeUrl + " and run it; this installer checks its digital signature first.")
    } else {
        Info ("Official download: " + $OllamaExeUrl + "   (this installer checks its digital signature first)")
    }
    if ($WithOllama -or (Ask-User "Install Ollama now?" "y")) {
        if (Have "winget") {
            $code = Invoke-Shown "winget" @("install", "--id", "Ollama.Ollama", "-e", "--accept-source-agreements", "--accept-package-agreements")
            if ($code -eq 0) { $viaWinget = $true } else { Warn "winget did not finish the install; trying the direct download." }
        }
        if (-not $viaWinget) {
            $exe = if ($script:Tmp) { Join-Path $script:Tmp "OllamaSetup.exe" } else { Join-Path ([IO.Path]::GetTempPath()) "OllamaSetup.exe" }
            Get-Download $OllamaExeUrl $exe
            $run = $true
            if (-not $DryRun) {
                Info ("Downloaded {0}, sha256 {1}" -f (Format-Size ((Get-Item -LiteralPath $exe).Length / 1MB)), (Get-FileSha256 $exe))
                Info "(the vendor does not publish a checksum list for this file; the digital signature below is the check.)"
                $sig = Get-AuthenticodeSignature -LiteralPath $exe
                if ($sig.Status -eq "Valid") { Ok ("Digital signature is valid. Signed by: " + $sig.SignerCertificate.Subject) }
                else {
                    Warn ("The digital signature check says: " + $sig.Status)
                    $run = Ask-User "Run the installer anyway? (not recommended)" "n"
                }
            }
            if ($run) {
                Info "Starting the Ollama installer; follow its window. No administrator rights are needed."
                if (-not $DryRun) { Say ("   $ " + $exe) "DarkGray"; Start-Process -FilePath $exe -Wait }
                else { Say ("   $ " + $exe + "  (dry run: not executed)") "DarkGray" }
            }
        }
        Update-SessionPath
        if (Have "ollama") { Ok "Ollama installed"; $ollamaReady = $true }
        elseif (-not $DryRun) {
            Warn "Ollama seems installed, but this window cannot see it yet."
            Info "Open a NEW PowerShell window and run this installer again; it picks up where it stopped."
        }
    } else {
        Info ("No problem. Install it yourself from " + $OllamaDlPage + " and run this installer again.")
    }
}
if ($ollamaReady -and -not $DryRun) {
    $up = Invoke-Quiet "ollama" @("list")
    if ($up.Code -ne 0) {
        Warn "The Ollama server is not answering."
        if (Ask-User "Start it now in the background (ollama serve)?" "y") {
            Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
            foreach ($i in 1..15) { Start-Sleep -Seconds 1; if ((Invoke-Quiet "ollama" @("list")).Code -eq 0) { break } }
            if ((Invoke-Quiet "ollama" @("list")).Code -eq 0) { Ok "Ollama is running" } else { Warn "Still not answering. Start the Ollama app from the Start menu, or run 'ollama serve' in another window." }
        }
    }
}

# ---------------------------------------------------------------- memory (embedding) models
# MIRRORS agent/embedmodels.py (tests/test_installers.py checks that tags and sizes agree). Number, tag, download MB, who it is for.
$EmbTable = @(
    @(1, "all-minilm", 46, "English only - smallest, fastest"),
    @(2, "nomic-embed-text", 274, "English only - small, good all-rounder"),
    @(3, "embeddinggemma:300m-qat-q4_0", 239, "Multilingual (100+ languages, e.g. German, French, Spanish) - small"),
    @(4, "granite-embedding:278m", 563, "Multilingual - mid size"),
    @(5, "bge-m3", 1200, "Multilingual (100+ languages, e.g. German, French) - best for non-English chats, biggest")
)

# english | multilingual | unknown, from the Windows display language (the same rule as agent/embedmodels.py language_hint)
function Get-EmbLang {
    $name = ""
    try { $name = (Get-Culture).Name } catch { }
    if ($env:LC_ALL) { $name = $env:LC_ALL } elseif ($env:LANG) { $name = $env:LANG }
    if (-not $name -or $name -match '^(C|POSIX)(\.|$)') { return "unknown" }
    if ($name -match '^en([-_.@]|$)') { return "english" }
    return "multilingual"
}

function Select-EmbedModel([int]$RamMb, [string]$Lang) {
    $small = "embeddinggemma:300m-qat-q4_0"
    if ($Lang -eq "english") { if ($RamMb -lt 4096) { return "all-minilm" } else { return "nomic-embed-text" } }
    if ($RamMb -lt 4096) { if ($Lang -eq "unknown") { return "all-minilm" } else { return $small } }
    if ($Lang -eq "multilingual" -and $RamMb -ge 16384) { return "bge-m3" }
    return $small
}

# Returns the chosen tag, or "" for none. A numbered choice; Enter (or -Yes) takes the recommended one.
function Select-EmbedChoice([int]$RamMb) {
    $rec = Select-EmbedModel $RamMb (Get-EmbLang)
    if ($Embed) { Info ("Using the memory model you asked for: " + $Embed); return $Embed }
    Say "   The memory model lets Nightage recall things by meaning. It runs on the CPU, so a smaller one leaves more memory for the chat model." "DarkGray"
    foreach ($row in $EmbTable) {
        $mark = if ($row[1] -eq $rec) { "   <- recommended for this computer" } else { "" }
        Say ("     {0}) {1,-32} {2,8}   {3}{4}" -f $row[0], $row[1], (Format-Size $row[2]), $row[3], $mark)
    }
    Say "     0) none - Nightage still works and searches memory by keywords"
    if ($Yes -or $DryRun) {
        $note = if ($DryRun -and -not $Yes) { "dry run: would ask, default" } else { "auto" }
        Say ("   ? Which memory model? [{0}: {1}]" -f $note, $rec) "Cyan"
        return $rec
    }
    $interactive = $true
    try { if ([Console]::IsInputRedirected) { $interactive = $false } } catch { }
    if (-not $interactive) {
        Stop-Install "There is no terminal to ask you on. Run again with -Yes to accept the defaults, or download the script and run it in a PowerShell window."
    }
    $reply = Read-Host "   ? Which memory model? [number, Enter = recommended]"
    if ($null -eq $reply) { $reply = "" }
    if ($reply -match '^\s*0' -or $reply -match '^\s*[nN]') { return "" }
    if ($reply -match '^\s*([1-5])\s*$') {
        $want = [int]$Matches[1]
        foreach ($row in $EmbTable) { if ($row[0] -eq $want) { return [string]$row[1] } }
    }
    return $rec
}

# ---- 6. model
Show-Step "The AI model"
if ($NoModel) {
    Info "Skipped (-NoModel). Pick one later with: ollama pull <name>   then   nightage models use <name>"
} else {
    $pick = Select-Model $ram $gpu.VramMb
    $chosen = $pick.Model; $sizeMb = $pick.SizeMb
    if ($Model) { $chosen = $Model; $sizeMb = 0; Info ("Using the model you asked for: " + $chosen) }
    else { Info ("For this computer (about {0} usable for a model, from {1}) the best fit is: {2}  (download about {3})" -f (Format-Size $pick.BudgetMb), $pick.Source, $chosen, (Format-Size $sizeMb)) }
    if ($NoOllama -or ((-not (Have "ollama")) -and (-not $DryRun))) {
        Info "Ollama is not available here, so nothing is downloaded now. Run this installer again once Ollama is installed,"
        Info ("or later:  ollama pull " + $chosen + "   then   nightage models use " + $chosen)
    } else {
        $have = (-not $DryRun) -and (Test-ModelInstalled $chosen)
        if ($have) { Ok ($chosen + " is already downloaded") }
        else {
            if (Ask-User ("Download " + $chosen + " now (one time)?") "y") {
                $code = Invoke-Shown "ollama" @("pull", $chosen)
                if ($code -ne 0) { Warn ("Download failed. Check your internet and run:  ollama pull " + $chosen) }
            } else { Info ("Skipped. Later:  ollama pull " + $chosen) }
        }
        if ($DryRun -or (Test-ModelInstalled $chosen)) {
            if (Ask-User ("Tell Nightage to use " + $chosen + " for everything (chat, code, notes)?") "y") {
                $code = Invoke-Shown $nightageExe @("models", "use", $chosen, "--yes")
                if ($code -ne 0) { Warn ("Could not save it now; run later:  nightage models use " + $chosen) }
            }
        }
        # the memory model: recall by meaning instead of by keywords (Nightage works without it)
        if ($NoEmbed) {
            Info "Skipped (-NoEmbed). Nightage searches memory by keywords. Later:  nightage embed list"
        } else {
            $emb = Select-EmbedChoice $ram
            if (-not $emb) {
                Info "No memory model. Nightage still works and falls back to keyword search. Later:  nightage embed list"
            } else {
                $embSize = 0
                foreach ($row in $EmbTable) { if ($row[1] -eq $emb) { $embSize = $row[2] } }
                $sizeText = if ($embSize -gt 0) { " (about " + (Format-Size $embSize) + ")" } else { "" }
                $embReady = $false
                if ((-not $DryRun) -and (Test-ModelInstalled $emb)) { Ok ($emb + " (the memory model) is already downloaded"); $embReady = $true }
                elseif (Ask-User ("Download the memory model " + $emb + $sizeText + "?") "y") {
                    $code = Invoke-Shown "ollama" @("pull", $emb)
                    if ($code -ne 0) { Warn ("Download failed. Later:  nightage embed use " + $emb) } else { $embReady = $true }
                } else { Info ("Skipped. Nightage still works and falls back to keyword search. Later:  nightage embed use " + $emb) }
                # `embed use` also rebuilds the memory index when the model changes, so an older install never mixes vector sizes
                if ($embReady -or $DryRun) {
                    $code = Invoke-Shown $nightageExe @("embed", "use", $emb, "--yes", "--no-pull")
                    if ($code -ne 0) { Warn ("Could not switch now; run later:  nightage embed use " + $emb) }
                }
            }
        }
    }
}

# ---- 7. finish
Show-Step "Checking everything"
if ($DryRun) { Info ("Dry run: would run  " + $nightageExe + " doctor --no-web --only system,python,packages") }
else {
    $code = Invoke-Shown $nightageExe @("doctor", "--no-web", "--only", "system,python,packages")
    if ($code -ne 0) { Warn "The check reported something; the lines above say what. Run 'nightage doctor' for the full report." }
}
Say ""
Say " All done." "Green"
Say "   What to type next (open a NEW PowerShell window first if the command is not found):"
Say "     nightage          start here: the home screen" "Cyan"
Say "     nightage demo     see what it looks like, no model needed" "Cyan"
Say "     nightage chat     start talking" "Cyan"
Say ""
Say ("   To update later: nightage update       To remove: powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall   (or: uv tool uninstall " + $ToolName + ")") "DarkGray"
if (-not $DryRun) { Say ("   Install log: " + $LogFile) "DarkGray" }
Remove-Temp
Stop-Log
