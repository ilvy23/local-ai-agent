"""`nightage night install-unit`: a systemd USER service that brings the night back after a crash or a reboot. It runs the same command with `--resume` and ends by itself when
the night is over (`.night/DONE` exists: the unit's condition stops being true). Exit 3 (a night that ended with unfinished prompts) counts as a normal end, exit 1 (refused: a night is already running, bad folder) and 130 (stopped) are never restarted; a crash or a kill is. Nothing is enabled or started by this module: it only renders the file; the CLI writes it and prints
the two commands to enable it (`systemctl --user enable --now NAME` and, so that it survives a reboot without a login, `loginctl enable-linger`)."""
from __future__ import annotations

import re
import shlex
from pathlib import Path


def name_for(root: Path) -> str:
    return "nightage-night-" + (re.sub(r"[^a-z0-9]+", "-", root.name.lower()).strip("-") or "project")


def unit_path(root: Path, config_home: Path | None = None) -> Path:
    return (config_home or Path.home() / ".config") / "systemd" / "user" / f"{name_for(root)}.service"


def render(root: Path, prompts: list[str], executable: str, extra_args: list[str] | None = None, path_env: str = "") -> str:
    cmd = " ".join(shlex.quote(a) for a in [executable, "night", "run", *prompts, "--folder", str(root), "--resume", "--yes", "--no-model", *(extra_args or [])])
    return f"""[Unit]
Description=nightage overnight coding run for {root.name}
After=default.target ollama-vk.service
ConditionPathExists=!{root}/.night/DONE
StartLimitIntervalSec=3600
StartLimitBurst=10

[Service]
Type=notify
NotifyAccess=main
WatchdogSec=3900
TimeoutStartSec=900
WorkingDirectory={root}
{"Environment=" + shlex.quote("PATH=" + path_env) if path_env else ""}
ExecStart={cmd}
Restart=on-failure
RestartSec=60
SuccessExitStatus=3
RestartPreventExitStatus=1 130
Nice=5
ExecStopPost=/bin/sh -c 'mkdir -p "{root}/.night"; echo "$(date -Is) result=$SERVICE_RESULT exit=$EXIT_CODE/$EXIT_STATUS" >> "{root}/.night/unit-exits.log"'

[Install]
WantedBy=default.target
"""


def instructions(root: Path) -> str:
    n = name_for(root)
    return (f"systemctl --user daemon-reload\nsystemctl --user enable --now {n}.service      # starts now and after every login\n"
            "loginctl enable-linger $USER                      # optional: also after a reboot without logging in\n"
            f"systemctl --user disable --now {n}.service       # when you do not want it any more (or delete the file)")
