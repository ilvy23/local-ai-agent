"""`python -m agent.doctor ...`: the doctor with every option, in a fresh process (used by `nightage update` to check the new code)."""

import typer

from agent.doctor.cli import register

app = typer.Typer(add_completion=False)
register(app)

if __name__ == "__main__":
    app()
