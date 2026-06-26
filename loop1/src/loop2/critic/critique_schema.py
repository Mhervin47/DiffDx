from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.7.0"


class TurnCritique(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    turn: int
    question_quality_score: float = Field(ge=0.0, le=1.0)
    differential_quality_score: float = Field(ge=0.0, le=1.0)
    reasoning_quality_score: float = Field(ge=0.0, le=1.0)
    confidence_calibration: Literal[
        "well-calibrated", "overconfident", "underconfident"
    ] | None = None
    weakness: str | None = None
    weakness_category: Literal[
        "redundant_question",
        "missed_red_flag",
        "poor_differential",
        "premature_confidence",
        "other",
    ] | None = None
    would_have_asked: str | None = None
    rationale: str
    critic_model: str
