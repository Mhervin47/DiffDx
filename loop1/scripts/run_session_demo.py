"""
Demo session runner — uses random exemplar retrieval instead of MMR so there is
no ML model warmup.  Startup is instant; suitable for live demos and quick tests.

For full MMR retrieval (Phase 5 production mode) use run_session.py instead.

Usage:
    python scripts/run_session_demo.py --profile test_cases/case_01.json
"""
from __future__ import annotations

import sys

# Print immediately before any other imports so the user sees output right away.
# On macOS Sequoia the OS security scanner can add ~60s to the first import of
# rich/typer in a new process; this line confirms the script is alive.
sys.stdout.write("MedicalConvo demo — loading libraries (first run may take ~60s on macOS)…\n")
sys.stdout.flush()

import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from rich.console import Console

import loop1.doctor as _doctor_module
from loop1.retrieval import select_random, _get_pool
from loop1.schemas import PatientProfile
from loop1.session import Session

app = typer.Typer(add_completion=False)
console = Console()


def _random_retrieval(profile, n: int = 3, lambda_: float = 0.7):
    """Drop-in replacement for get_exemplars_for_profile with no ML warmup."""
    return select_random(_get_pool(), n=n)


@app.command()
def main(
    profile: Path = typer.Option(
        ..., "--profile", "-p", help="Path to a patient profile JSON file."
    ),
    max_turns: int = typer.Option(
        None, "--max-turns", "-n", help="Override max turns from config."
    ),
    confidence_threshold: float = typer.Option(
        None,
        "--threshold",
        "-c",
        help="Override confidence_to_stop threshold from config.",
    ),
) -> None:
    if not profile.exists():
        console.print(f"[red]Error:[/red] profile not found: {profile}", err=True)
        raise typer.Exit(1)

    raw = json.loads(profile.read_text(encoding="utf-8"))
    patient = PatientProfile(**raw)

    console.print(f"[bold]Profile :[/bold] {profile}")
    console.print(
        f"[bold]Patient :[/bold] "
        f"{patient.demographics.age}yo {patient.demographics.sex} — {patient.chief_complaint}"
    )
    console.print(f"[bold]Session :[/bold] {patient.session_id}")
    console.print("[dim]Retrieval: random (demo mode — no ML warmup)[/dim]\n")
    console.print("Type [bold]quit[/bold] at any prompt to end the session early.\n")

    # Patch doctor module to use random retrieval — no sentence_transformers/FAISS needed
    _doctor_module.get_exemplars_for_profile = _random_retrieval

    kwargs: dict = {}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold

    session = Session(patient, **kwargs)
    session.run()


if __name__ == "__main__":
    app()
