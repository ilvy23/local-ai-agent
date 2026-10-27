"""`nightage doctor` and first-run onboarding: the stable import point.

The implementation lives in the modular package `agent/doctor/` (checks, runner, fixes, report, CLI; guide: docs/DOCTOR.md). This
module keeps the names the home screen, main.py and the tests import:

  run_doctor, collect_checks, needs_onboarding, run_onboarding, suggest_model, render_report   the cheap core set (ok / warn / fail)
  DoctorReport, Check, Probes, OllamaInfo, model_installed

The full command (`--only`, `--deep`, `--fix`, `--report`, ...) is `agent.doctor.cli.doctor_command`.
"""

from __future__ import annotations

from agent.doctor.legacy import (  # noqa: F401
    EXAMPLE_PROMPTS,
    FAIL,
    OK,
    REQUIRED_PACKAGES,
    WARN,
    Check,
    DoctorReport,
    collect_checks,
    needs_onboarding,
    render_report,
    run_doctor,
    run_onboarding,
    suggest_model,
)
from agent.doctor.probes import OllamaInfo, Probes, model_installed  # noqa: F401
