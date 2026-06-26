from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.9.0"
ROUTING_MODEL_VERSION = "disease_specialty_map_v1"


class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"


class AppointmentType(str, Enum):
    EMERGENCY = "emergency_visit"
    SAME_DAY = "same_day_appointment"
    URGENT = "urgent_appointment_within_48h"
    ROUTINE = "routine_appointment"
    IN_PERSON = "in_person"
    TELEHEALTH = "telehealth_consultation"


class RoutingOption(BaseModel):
    specialty: str
    urgency: UrgencyLevel
    reasoning: str
    appointment_type: AppointmentType
    confidence_weight: float = Field(ge=0.0, le=1.0)


class RoutingDecision(BaseModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    patient_id: Optional[str] = None
    is_ambiguous: bool
    primary_routing: RoutingOption
    alternative_routings: list[RoutingOption] = Field(default_factory=list)
    final_differential: list[tuple[str, float]]
    final_confidence: float
    routing_model_version: str = ROUTING_MODEL_VERSION
