"""Run a full diagnostic dialogue session with manual patient input."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from rich.console import Console

from loop1.retrieval import _get_embed_model, _get_pool, _get_index
from loop1.schemas import PatientProfile
from loop1.session import Session

app = typer.Typer(add_completion=False)
console = Console()


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
    console.print(f"[bold]Session :[/bold] {patient.session_id}\n")
    console.print("Type [bold]quit[/bold] at any prompt to end the session early.\n")

    with console.status("[dim]Loading retrieval model…[/dim]"):
        _get_embed_model()
        _get_pool()
        _get_index()

    kwargs: dict = {}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold

    session = Session(patient, **kwargs)
    session.run()


if __name__ == "__main__":
    app()
