"""
Demo: single interactive diagnostic session with real-time turn-level critique.

Case: case_01.json — 58M, chest pain + shortness of breath 2 hrs (ACS).

Usage:
    cd /Users/mhervin/Desktop/MedicalConvo/loop1
    PYTHONPATH=src .venv/bin/python3 demo_with_critic.py
"""
import json
import os
import sys
import uuid
from pathlib import Path

# ── env ───────────────────────────────────────────────────────────────────────
try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

os.environ["CRITIC_MODEL"] = "groq/llama-3.3-70b-versatile"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.table import Table
from rich import box

from loop1.schemas import PatientProfile, TurnRecord
from loop1.session import Session
from loop1.logging_utils import log_event
from loop2.critic.critic import critique_turn
from loop2.critic.critique_schema import TurnCritique, SCHEMA_VERSION
from loop2.critic.aggregator import aggregate_critiques


def _write_critic_report(
    session_id: str,
    profile: PatientProfile,
    final_record,
    critiques: list[TurnCritique],
    agg: dict,
) -> Path:
    report = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "patient": {
            "age": profile.demographics.age,
            "sex": profile.demographics.sex,
            "chief_complaint": profile.chief_complaint,
        },
        "final_diagnosis": final_record.primary_diagnosis,
        "termination_reason": final_record.termination_reason,
        "total_turns": len(critiques),
        "aggregate": agg,
        "turns": [c.model_dump() for c in critiques],
    }
    out_dir = Path(__file__).parent / "logs" / "critic_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"critic_{session_id}.json"
    path.write_text(json.dumps(report, indent=2))
    return path

console = Console()


class LiveCriticSession(Session):
    """Session subclass that critiques each turn immediately after logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._live_events: list[dict] = []
        self._critiques: list[TurnCritique] = []

    def _log_turn(self, turn_record: TurnRecord, prompt_tokens: int) -> None:
        # Build the same event dict that log_event writes to JSONL
        event = {
            "event_type": "turn_complete",
            "session_id": self.profile.session_id,
            "turn_index": turn_record.turn_index,
            "doctor_output": turn_record.doctor_output.model_dump(),
            "patient_answer": turn_record.patient_answer,
            "profile_state": self.profile.model_dump(),
            "prompt_tokens": prompt_tokens,
            "timestamp": turn_record.timestamp,
        }
        self._live_events.append(event)

        # Write to disk as normal
        super()._log_turn(turn_record, prompt_tokens)

        # Critique this turn immediately
        try:
            critique = critique_turn(event, self._live_events, self.profile.session_id)
            self._critiques.append(critique)
            self._print_live_critique(critique)
        except Exception as exc:
            console.print(f"  [dim red]Critic skipped turn {turn_record.turn_index}: {exc}[/dim red]")

    def _print_live_critique(self, c: TurnCritique) -> None:
        q_col = "green" if c.question_quality_score >= 0.7 else ("yellow" if c.question_quality_score >= 0.5 else "red")
        d_col = "green" if c.differential_quality_score >= 0.7 else ("yellow" if c.differential_quality_score >= 0.5 else "red")
        weakness = f"[yellow]{c.weakness_category}[/yellow]" if c.weakness_category and c.weakness_category != "other" else (c.weakness_category or "—")
        console.print(
            f"  [dim]↳ critic[/dim]  "
            f"Q=[{q_col}]{c.question_quality_score:.2f}[/{q_col}]  "
            f"Diff=[{d_col}]{c.differential_quality_score:.2f}[/{d_col}]  "
            f"Reasoning={c.reasoning_quality_score:.2f}  "
            f"{weakness}"
        )
        if c.would_have_asked and c.question_quality_score < 0.7:
            console.print(f"  [dim]  would have asked: {c.would_have_asked[:80]}[/dim]")
        console.print()


# ── patient ───────────────────────────────────────────────────────────────────
CASE = Path(__file__).parent / "test_cases" / "case_01.json"
profile = PatientProfile.model_validate(json.loads(CASE.read_text()))
profile.session_id = str(uuid.uuid4())

console.rule("[bold]Demo Session — Case 01[/bold]")
console.print(f"  Patient         : 58M, accountant, BMI 31")
console.print(f"  Chief complaint : {profile.chief_complaint}")
console.rule()

# ── run session with live critique ────────────────────────────────────────────
session = LiveCriticSession(profile, max_turns=6)
final_record = session.run()

# ── final summary table ───────────────────────────────────────────────────────
if session._critiques:
    console.rule("[bold cyan]Full session critique[/bold cyan]")

    tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    tbl.add_column("Turn", style="dim", width=5)
    tbl.add_column("Q-quality", justify="center", width=9)
    tbl.add_column("Diff-quality", justify="center", width=12)
    tbl.add_column("Reasoning", justify="center", width=10)
    tbl.add_column("Calibration", width=16)
    tbl.add_column("Weakness", width=20)
    tbl.add_column("Would have asked", width=40)

    for c in session._critiques:
        q_col = "green" if c.question_quality_score >= 0.7 else ("yellow" if c.question_quality_score >= 0.5 else "red")
        d_col = "green" if c.differential_quality_score >= 0.7 else ("yellow" if c.differential_quality_score >= 0.5 else "red")
        r_col = "green" if c.reasoning_quality_score >= 0.7 else ("yellow" if c.reasoning_quality_score >= 0.5 else "red")
        tbl.add_row(
            str(c.turn + 1),
            f"[{q_col}]{c.question_quality_score:.2f}[/{q_col}]",
            f"[{d_col}]{c.differential_quality_score:.2f}[/{d_col}]",
            f"[{r_col}]{c.reasoning_quality_score:.2f}[/{r_col}]",
            c.confidence_calibration or "—",
            c.weakness_category or "—",
            (c.would_have_asked or "—")[:38],
        )

    console.print(tbl)

    agg = aggregate_critiques(session._critiques)
    console.rule("[bold cyan]Summary[/bold cyan]")
    console.print(f"  Mean Q-quality       : {agg['mean_question_quality']}")
    console.print(f"  Mean diff-quality    : {agg['mean_differential_quality']}")
    console.print(f"  Mean reasoning       : {agg['mean_reasoning_quality']}")
    console.print(f"  Calibration spread   : {agg['confidence_calibration_distribution']}")
    console.print(f"  Weakness categories  : {agg['weakness_category_counts']}")

    report_path = _write_critic_report(
        final_record.session_id, profile, final_record, session._critiques, agg
    )
    console.print(f"\n  Critic report → [bold]{report_path}[/bold]")

console.print()
console.print(f"  Final diagnosis      : [bold green]{final_record.primary_diagnosis}[/bold green]")
console.print(f"  Termination reason   : {final_record.termination_reason}")
console.rule()
