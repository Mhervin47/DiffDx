from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from loop2.critic.critique_schema import TurnCritique


def load_critiques(critique_jsonl_path: str | Path) -> list[TurnCritique]:
    """Load TurnCritique records from a JSONL file."""
    path = Path(critique_jsonl_path)
    critiques: list[TurnCritique] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                critiques.append(TurnCritique.model_validate(data))
    return critiques


def aggregate_critiques(critiques: list[TurnCritique]) -> dict[str, Any]:
    """
    Roll up critique JSONL records into summary statistics.

    Returns a dict with:
        - mean/median question/differential/reasoning quality
        - confidence_calibration distribution
        - weakness_category counts
        - sessions with low question quality (<0.5) per session_id
        - most common weakness texts (top 5)
    """
    if not critiques:
        return {
            "total_turns": 0,
            "mean_question_quality": None,
            "median_question_quality": None,
            "mean_differential_quality": None,
            "median_differential_quality": None,
            "mean_reasoning_quality": None,
            "median_reasoning_quality": None,
            "confidence_calibration_distribution": {},
            "weakness_category_counts": {},
            "low_question_quality_sessions": [],
            "top_weakness_texts": [],
        }

    q_scores = [c.question_quality_score for c in critiques]
    d_scores = [c.differential_quality_score for c in critiques]
    r_scores = [c.reasoning_quality_score for c in critiques]

    calibration_dist = Counter(
        c.confidence_calibration for c in critiques if c.confidence_calibration is not None
    )
    category_counts = Counter(
        c.weakness_category for c in critiques if c.weakness_category is not None
    )

    # Sessions where at least one turn has question_quality < 0.5
    low_q_sessions = sorted(set(
        c.session_id for c in critiques if c.question_quality_score < 0.5
    ))

    # Top 5 weakness texts by frequency
    weakness_counter = Counter(
        c.weakness for c in critiques if c.weakness
    )
    top_weaknesses = [{"weakness": w, "count": n} for w, n in weakness_counter.most_common(5)]

    return {
        "total_turns": len(critiques),
        "total_sessions": len(set(c.session_id for c in critiques)),
        "mean_question_quality": round(statistics.mean(q_scores), 3),
        "median_question_quality": round(statistics.median(q_scores), 3),
        "mean_differential_quality": round(statistics.mean(d_scores), 3),
        "median_differential_quality": round(statistics.median(d_scores), 3),
        "mean_reasoning_quality": round(statistics.mean(r_scores), 3),
        "median_reasoning_quality": round(statistics.median(r_scores), 3),
        "confidence_calibration_distribution": dict(calibration_dist),
        "weakness_category_counts": dict(category_counts),
        "low_question_quality_sessions": low_q_sessions,
        "top_weakness_texts": top_weaknesses,
    }


def cross_reference_failures(
    critiques: list[TurnCritique],
    eval_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Cross-reference sessions with low question quality AND wrong diagnosis.

    Args:
        critiques: All TurnCritique records across sessions.
        eval_results: List of DDXPlusEvalResult dicts (or model_dump() output).

    Returns:
        List of dicts, one per session, with question quality stats and diagnostic outcome.
    """
    # Group critiques by session_id
    by_session: dict[str, list[TurnCritique]] = {}
    for c in critiques:
        by_session.setdefault(c.session_id, []).append(c)

    # Build eval lookup by session_id
    eval_by_session: dict[str, dict] = {
        e["session_id"]: e for e in eval_results
    }

    rows: list[dict[str, Any]] = []
    for session_id, sess_critiques in by_session.items():
        eval_rec = eval_by_session.get(session_id, {})
        q_scores = [c.question_quality_score for c in sess_critiques]
        mean_q = statistics.mean(q_scores) if q_scores else None
        low_quality_turns = [c.turn for c in sess_critiques if c.question_quality_score < 0.5]
        categories = [c.weakness_category for c in sess_critiques if c.weakness_category]

        rows.append({
            "session_id": session_id,
            "patient_id": eval_rec.get("patient_id", "unknown"),
            "mean_question_quality": round(mean_q, 3) if mean_q is not None else None,
            "low_quality_turns": low_quality_turns,
            "leading_diagnosis_correct": eval_rec.get("leading_diagnosis_correct"),
            "top3_contains_truth": eval_rec.get("top3_contains_truth"),
            "in_exemplar_pool": eval_rec.get("in_exemplar_pool"),
            "total_turns": len(sess_critiques),
            "weakness_categories": categories,
        })

    # Sort: wrong diagnosis first, then by mean question quality ascending
    rows.sort(key=lambda r: (
        0 if r["leading_diagnosis_correct"] is False else 1,
        r["mean_question_quality"] or 1.0,
    ))
    return rows
