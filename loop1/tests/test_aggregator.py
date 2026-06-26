from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from loop2.critic.aggregator import aggregate_critiques, cross_reference_failures, load_critiques
from loop2.critic.critique_schema import TurnCritique


def _make_critique(
    session_id: str = "s1",
    turn: int = 0,
    q: float = 0.8,
    d: float = 0.7,
    r: float = 0.9,
    calibration: str | None = "well-calibrated",
    weakness_category: str | None = None,
    weakness: str | None = None,
) -> TurnCritique:
    return TurnCritique(
        session_id=session_id,
        turn=turn,
        question_quality_score=q,
        differential_quality_score=d,
        reasoning_quality_score=r,
        confidence_calibration=calibration,  # type: ignore[arg-type]
        weakness=weakness,
        weakness_category=weakness_category,  # type: ignore[arg-type]
        would_have_asked=None,
        rationale="Test rationale.",
        critic_model="test",
    )


def _write_critique_jsonl(critiques: list[TurnCritique]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for c in critiques:
        tmp.write(c.model_dump_json() + "\n")
    tmp.flush()
    return Path(tmp.name)


class TestLoadCritiques:
    def test_roundtrip(self):
        critiques = [
            _make_critique("s1", 0, q=0.7),
            _make_critique("s1", 1, q=0.4, weakness_category="redundant_question"),
        ]
        path = _write_critique_jsonl(critiques)
        loaded = load_critiques(path)
        assert len(loaded) == 2
        assert loaded[0].question_quality_score == pytest.approx(0.7)
        assert loaded[1].weakness_category == "redundant_question"


class TestAggregateCritiques:
    def test_empty(self):
        result = aggregate_critiques([])
        assert result["total_turns"] == 0
        assert result["mean_question_quality"] is None

    def test_basic_stats(self):
        critiques = [
            _make_critique("s1", 0, q=0.8, d=0.9, r=0.7),
            _make_critique("s1", 1, q=0.4, d=0.5, r=0.6),
            _make_critique("s2", 0, q=0.6, d=0.7, r=0.8),
        ]
        result = aggregate_critiques(critiques)

        assert result["total_turns"] == 3
        assert result["total_sessions"] == 2
        assert result["mean_question_quality"] == pytest.approx((0.8 + 0.4 + 0.6) / 3, abs=0.001)
        assert result["median_question_quality"] == pytest.approx(0.6, abs=0.001)

    def test_calibration_distribution(self):
        critiques = [
            _make_critique("s1", 0, calibration="well-calibrated"),
            _make_critique("s1", 1, calibration="overconfident"),
            _make_critique("s1", 2, calibration="overconfident"),
            _make_critique("s1", 3, calibration=None),
        ]
        result = aggregate_critiques(critiques)
        dist = result["confidence_calibration_distribution"]
        assert dist["overconfident"] == 2
        assert dist["well-calibrated"] == 1
        assert "None" not in dist

    def test_weakness_category_counts(self):
        critiques = [
            _make_critique("s1", 0, weakness_category="redundant_question"),
            _make_critique("s1", 1, weakness_category="missed_red_flag"),
            _make_critique("s1", 2, weakness_category="redundant_question"),
            _make_critique("s1", 3, weakness_category=None),
        ]
        result = aggregate_critiques(critiques)
        counts = result["weakness_category_counts"]
        assert counts["redundant_question"] == 2
        assert counts["missed_red_flag"] == 1

    def test_low_quality_sessions(self):
        critiques = [
            _make_critique("s1", 0, q=0.3),   # low quality
            _make_critique("s1", 1, q=0.8),
            _make_critique("s2", 0, q=0.9),   # all good
            _make_critique("s3", 0, q=0.2),   # low quality
        ]
        result = aggregate_critiques(critiques)
        low_sessions = result["low_question_quality_sessions"]
        assert "s1" in low_sessions
        assert "s3" in low_sessions
        assert "s2" not in low_sessions

    def test_top_weakness_texts(self):
        critiques = [
            _make_critique("s1", 0, weakness="Asked about fever again"),
            _make_critique("s1", 1, weakness="Asked about fever again"),
            _make_critique("s1", 2, weakness="Missed dyspnea screening"),
            _make_critique("s2", 0, weakness="Asked about fever again"),
        ]
        result = aggregate_critiques(critiques)
        top = result["top_weakness_texts"]
        assert top[0]["weakness"] == "Asked about fever again"
        assert top[0]["count"] == 3


class TestCrossReferenceFailures:
    def _make_eval(self, session_id: str, correct: bool, patient_id: str = "p1", in_pool: bool = True) -> dict:
        return {
            "session_id": session_id,
            "patient_id": patient_id,
            "leading_diagnosis_correct": correct,
            "top3_contains_truth": correct,
            "in_exemplar_pool": in_pool,
        }

    def test_wrong_diagnosis_sorted_first(self):
        critiques = [
            _make_critique("s1", 0, q=0.8),
            _make_critique("s2", 0, q=0.3),
        ]
        evals = [
            self._make_eval("s1", correct=True),
            self._make_eval("s2", correct=False),
        ]
        rows = cross_reference_failures(critiques, evals)
        assert rows[0]["session_id"] == "s2"  # wrong diagnosis first

    def test_includes_session_without_eval(self):
        critiques = [_make_critique("s_no_eval", 0, q=0.5)]
        rows = cross_reference_failures(critiques, [])
        assert len(rows) == 1
        assert rows[0]["leading_diagnosis_correct"] is None

    def test_weakness_categories_aggregated(self):
        critiques = [
            _make_critique("s1", 0, weakness_category="redundant_question"),
            _make_critique("s1", 1, weakness_category="missed_red_flag"),
        ]
        rows = cross_reference_failures(critiques, [])
        assert "redundant_question" in rows[0]["weakness_categories"]
        assert "missed_red_flag" in rows[0]["weakness_categories"]
