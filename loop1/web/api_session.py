"""
APISession: step-based diagnostic session for the web API.

Unlike Session.run() which loops interactively, APISession exposes
two methods: initialize() and submit_answer(). Each returns a JSON-
serialisable dict with the doctor question, differential, and critique.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Load .env before importing loop1 modules
try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

os.environ.setdefault("CRITIC_MODEL", "openrouter/google/gemma-4-31b-it:free")

from loop1.closing_turn import generate_closing_turn
from loop1.compressor import compress_context
from loop1.config import config
from loop1.doctor import generate_turn_with_usage
from loop1.logging_utils import log_event, write_final_record
from loop1.safety import check_safety
from loop1.schemas import (
    DiagnosisEntry,
    DoctorTurnOutput,
    FinalRecord,
    ModelMetadata,
    PatientProfile,
    TurnRecord,
)
from loop2.critic.aggregator import aggregate_critiques
from loop2.critic.critic import critique_turn
from loop2.critic.critique_schema import TurnCritique

_log = logging.getLogger(__name__)


class APISession:
    """
    Manages one diagnostic session driven turn-by-turn from the web API.

    Flow:
        session = APISession(profile)
        result = session.initialize()          # returns first doctor question
        result = session.submit_answer(text)   # returns next question + critique
        ...until result["session_complete"] is True
        report = session.get_report()
    """

    def __init__(self, profile: PatientProfile, max_turns: int = 6) -> None:
        self.profile = profile
        self.session_id = profile.session_id
        self.max_turns = max_turns
        self.confidence_threshold: float = config["thresholds"]["confidence_to_stop"]
        self.keep_recent: int = config["thresholds"]["compression_keep_recent"]

        self.history: list[TurnRecord] = []
        self._live_events: list[dict] = []
        self._critiques: list[TurnCritique] = []
        self.started_at = datetime.now(timezone.utc).isoformat()

        # Pending question state
        self._pending_turn_index: int = 0
        self._pending_doctor_output: DoctorTurnOutput | None = None
        self._pending_prompt_tokens: int = 0
        self._pending_exemplar_ids: list[str] = []

        self.complete: bool = False
        self.termination_reason: str | None = None
        self._final_record: FinalRecord | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> dict:
        """Generate the first doctor question. Call once after creation."""
        log_event(
            session_id=self.session_id,
            event_type="session_start",
            data={
                "session_id": self.session_id,
                "chief_complaint": self.profile.chief_complaint,
                "max_turns": self.max_turns,
                "confidence_threshold": self.confidence_threshold,
            },
        )
        doctor_output, prompt_tokens, exemplar_ids = generate_turn_with_usage(
            self.profile, self._recent_history(), 0
        )

        if doctor_output.should_stop or doctor_output.confidence_to_stop >= self.confidence_threshold:
            self._finalize("confidence_threshold", doctor_output)
            return self._build_response(doctor_output, None)

        self._set_pending(0, doctor_output, prompt_tokens, exemplar_ids)
        return self._build_response(doctor_output, None)

    def submit_answer(self, patient_answer: str) -> dict:
        """Log the patient answer, run critique, and generate the next question."""
        if self.complete:
            raise ValueError("Session is already complete.")
        if self._pending_doctor_output is None:
            raise ValueError("No pending question — call initialize() first.")

        doctor_output = self._pending_doctor_output
        turn_index = self._pending_turn_index

        # Safety check
        matched = check_safety(patient_answer)
        if matched:
            self._finalize("safety_stop", doctor_output)
            return self._build_response(doctor_output, None)

        # Build event dict (mirrors LiveCriticSession._log_turn)
        now = datetime.now(timezone.utc).isoformat()
        event = {
            "event_type": "turn_complete",
            "session_id": self.session_id,
            "turn_index": turn_index,
            "doctor_output": doctor_output.model_dump(),
            "patient_answer": patient_answer,
            "profile_state": self.profile.model_dump(),
            "prompt_tokens": self._pending_prompt_tokens,
            "timestamp": now,
        }
        self._live_events.append(event)

        turn_record = TurnRecord(
            turn_index=turn_index,
            doctor_output=doctor_output,
            patient_answer=patient_answer,
            retrieved_exemplar_ids=self._pending_exemplar_ids,
            timestamp=now,
        )
        log_event(
            session_id=self.session_id,
            event_type="turn_complete",
            data={
                "turn_index": turn_index,
                "doctor_output": doctor_output.model_dump(),
                "patient_answer": patient_answer,
                "retrieved_exemplar_ids": self._pending_exemplar_ids,
                "prompt_tokens": self._pending_prompt_tokens,
                "timestamp": now,
                "profile_state": self.profile.model_dump(),
            },
        )
        self.history.append(turn_record)

        # Fire critic in background — never wait for it. Scores appear in the final report.
        # OpenRouter/Gemma-4 free tier runs on a separate quota from Groq/Gemini.
        self._fire_critic(event, turn_index)

        # Profile updater disabled in web session — it makes an extra Groq call per turn
        # that consistently rate-limits subsequent doctor question calls on Groq's free tier.
        # The full turn history passed to the doctor already provides enough context.
        self._maybe_compress()

        next_index = turn_index + 1

        # Max turns reached — one more generation for the final differential, then close
        if next_index >= self.max_turns:
            final_output, _, _ = generate_turn_with_usage(
                self.profile, self._recent_history(), next_index
            )
            self._finalize("max_turns", final_output)
            return self._build_response(final_output, None)

        # Generate next question
        next_output, next_tokens, next_exemplar_ids = generate_turn_with_usage(
            self.profile, self._recent_history(), next_index
        )

        if next_output.should_stop or next_output.confidence_to_stop >= self.confidence_threshold:
            self._finalize("confidence_threshold", next_output)
            return self._build_response(next_output, None)

        self._set_pending(next_index, next_output, next_tokens, next_exemplar_ids)
        return self._build_response(next_output, None)

    def get_report(self) -> dict | None:
        """Return the full critic report JSON once the session is complete."""
        if not self._final_record:
            return None
        agg = aggregate_critiques(self._critiques) if self._critiques else {}
        closing = self._final_record.closing_turn
        return {
            "schema_version": "0.7.0",
            "session_id": self.session_id,
            "patient": {
                "age": self.profile.demographics.age,
                "sex": self.profile.demographics.sex,
                "chief_complaint": self.profile.chief_complaint,
            },
            "final_diagnosis": self._final_record.primary_diagnosis,
            "termination_reason": self._final_record.termination_reason,
            "total_turns": len(self._critiques),
            "aggregate": agg,
            "turns": [c.model_dump() for c in self._critiques],
            "closing_turn": closing.model_dump() if closing else None,
            "final_differential": [
                {"dx": d.dx, "prob": d.prob}
                for d in self._final_record.final_differential
            ],
            "turn_history": [
                {
                    "turn_index": tr.turn_index,
                    "question": tr.doctor_output.chosen_question,
                    "rationale": tr.doctor_output.rationale,
                    "patient_answer": tr.patient_answer,
                    "differential": [
                        {"dx": d.dx, "prob": d.prob}
                        for d in tr.doctor_output.current_differential
                    ],
                    "confidence": tr.doctor_output.confidence_to_stop,
                }
                for tr in self._final_record.turn_history
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recent_history(self) -> list[TurnRecord]:
        if self.profile.running_summary:
            return self.history[-self.keep_recent:]
        return self.history

    def _maybe_compress(self) -> None:
        if len(self.history) <= self.keep_recent:
            return
        self.profile = compress_context(self.profile, self.history, self.keep_recent)
        log_event(
            session_id=self.session_id,
            event_type="compression_complete",
            data={
                "turns_summarized": len(self.history) - self.keep_recent,
                "running_summary_length": len(self.profile.running_summary),
            },
        )

    def _set_pending(
        self,
        turn_index: int,
        doctor_output: DoctorTurnOutput,
        prompt_tokens: int,
        exemplar_ids: list[str],
    ) -> None:
        self._pending_turn_index = turn_index
        self._pending_doctor_output = doctor_output
        self._pending_prompt_tokens = prompt_tokens
        self._pending_exemplar_ids = exemplar_ids

    def _finalize(self, termination_reason: str, final_doctor_output: DoctorTurnOutput) -> None:
        self.complete = True
        self.termination_reason = termination_reason

        closing = None
        if termination_reason != "safety_stop":
            closing = generate_closing_turn(
                self.profile, final_doctor_output.current_differential
            )

        primary = (
            final_doctor_output.current_differential[0].dx
            if final_doctor_output.current_differential
            else "undetermined"
        )
        self._final_record = FinalRecord(
            session_id=self.session_id,
            started_at=self.started_at,
            ended_at=datetime.now(timezone.utc).isoformat(),
            termination_reason=termination_reason,  # type: ignore[arg-type]
            final_profile=self.profile,
            final_differential=final_doctor_output.current_differential,
            primary_diagnosis=primary,
            turn_history=self.history,
            model_metadata=ModelMetadata(
                model_name=config["models"]["doctor"],
                model_version="0",
                prompt_template_version=config["prompt_versions"]["doctor"],
            ),
            closing_turn=closing,
        )
        write_final_record(self.session_id, self._final_record.model_dump())
        log_event(
            session_id=self.session_id,
            event_type="session_end",
            data={
                "termination_reason": termination_reason,
                "doctor_output": final_doctor_output.model_dump(),
            },
        )

    def _fire_critic(self, event: dict, turn_index: int) -> None:
        """Submit a critique task and return immediately. Result stored in self._critiques."""
        def _run() -> None:
            try:
                result = critique_turn(event, self._live_events, self.session_id)
                if result:
                    self._critiques.append(result)
            except Exception as exc:
                _log.warning("Critic failed for turn %d: %s", turn_index, exc)

        try:
            pool = ThreadPoolExecutor(max_workers=1)
            pool.submit(_run)
            pool.shutdown(wait=False)
        except Exception as exc:
            _log.warning("Could not launch critic thread: %s", exc)

    def _build_response(
        self,
        doctor_output: DoctorTurnOutput,
        critique: TurnCritique | None,
    ) -> dict:
        critique_data = None
        if critique is not None:
            critique_data = {
                "question_quality_score": critique.question_quality_score,
                "differential_quality_score": critique.differential_quality_score,
                "reasoning_quality_score": critique.reasoning_quality_score,
                "confidence_calibration": critique.confidence_calibration,
                "weakness_category": critique.weakness_category,
                "would_have_asked": critique.would_have_asked,
                "rationale": critique.rationale,
            }
        return {
            "doctor_question": doctor_output.chosen_question,
            "differential": [
                {"dx": d.dx, "prob": d.prob}
                for d in doctor_output.current_differential
            ],
            "confidence_to_stop": doctor_output.confidence_to_stop,
            "session_complete": self.complete,
            "termination_reason": self.termination_reason,
            "critique": critique_data,
        }
