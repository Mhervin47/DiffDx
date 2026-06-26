"""
Eval script: compare select_random vs get_exemplars_for_profile (MMR) retrieval quality.

For each test case, measures target exemplar appearance rate at turn 1, 2, and 3.
MMR is deterministic so it is run once per turn; random is run N=20 times per turn.

No LLM calls needed — retrieval quality is measured directly.

Usage:
    PYTHONPATH=src python scripts/eval_retrieval.py
    PYTHONPATH=src python scripts/eval_retrieval.py --n 50
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loop1.retrieval import get_exemplars_for_profile, load_pool, select_random
from loop1.schemas import Demographics, History, PatientProfile, Symptom

_EXEMPLARS_DIR = Path(__file__).parent.parent / "exemplars"
N_DEFAULT = 20
TOP_K = 3
PASS_THRESHOLD = 0.80  # >80% hit rate required at turn 2 for key cases


# ---------------------------------------------------------------------------
# Test case definitions — profile at turn 1, 2, 3
# ---------------------------------------------------------------------------

@dataclass
class TurnSnapshot:
    chief_complaint: str
    symptoms: list[Symptom] = field(default_factory=list)
    running_summary: str = ""


@dataclass
class EvalCase:
    name: str
    target_exemplar_id: str
    turns: list[TurnSnapshot]  # index 0 = turn 1, 1 = turn 2, 2 = turn 3
    key_case: bool = False      # True → apply pass/fail criterion


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        name="Measles (9mo, cephalocaudal rash, unvaccinated, travel)",
        target_exemplar_id="ex_measles_001",
        key_case=True,
        turns=[
            TurnSnapshot(
                chief_complaint="fever and rash in a 9-month-old child",
            ),
            TurnSnapshot(
                chief_complaint="fever and rash in a 9-month-old child",
                symptoms=[
                    Symptom(name="high fever", onset="4 days ago"),
                    Symptom(name="blotchy red rash starting at hairline and spreading downward"),
                    Symptom(name="cough"),
                    Symptom(name="runny nose"),
                    Symptom(name="red watery eyes"),
                ],
            ),
            TurnSnapshot(
                chief_complaint="fever and rash in a 9-month-old child",
                symptoms=[
                    Symptom(name="high fever", onset="4 days ago"),
                    Symptom(name="blotchy red rash starting at hairline and spreading downward"),
                    Symptom(name="cough"),
                    Symptom(name="runny nose"),
                    Symptom(name="red watery eyes"),
                ],
                running_summary=(
                    "9-month-old child, unvaccinated (below routine MMR age), returned from "
                    "international travel 10 days ago. Cephalocaudal rash progression following "
                    "3-day prodrome of cough, coryza, and conjunctivitis. Feeding poorly."
                ),
            ),
        ],
    ),
    EvalCase(
        name="CVST (26yo F, OCP, progressive severe headache)",
        target_exemplar_id="ex_cvst_001",
        key_case=True,
        turns=[
            TurnSnapshot(
                chief_complaint="severe headache in a young woman on the contraceptive pill",
            ),
            TurnSnapshot(
                chief_complaint="severe headache in a young woman on the contraceptive pill",
                symptoms=[
                    Symptom(name="thunderclap headache", onset="3 days ago", severity="severe"),
                    Symptom(name="worsening when lying down"),
                    Symptom(name="nausea"),
                    Symptom(name="blurred vision"),
                ],
            ),
            TurnSnapshot(
                chief_complaint="severe headache in a young woman on the contraceptive pill",
                symptoms=[
                    Symptom(name="thunderclap headache", onset="3 days ago", severity="severe"),
                    Symptom(name="worsening when lying down"),
                    Symptom(name="nausea"),
                    Symptom(name="blurred vision"),
                ],
                running_summary=(
                    "26-year-old woman on OCP for 6 months. Progressive severe headache worse "
                    "on lying flat. Neurological symptoms raising concern for cerebral venous "
                    "sinus thrombosis. No focal weakness yet."
                ),
            ),
        ],
    ),
    EvalCase(
        name="Pulmonary embolism (pleuritic chest pain, recent flight)",
        target_exemplar_id="ex_pulmonary_embolism_001",
        turns=[
            TurnSnapshot(
                chief_complaint="sudden chest pain and shortness of breath after a long flight",
            ),
            TurnSnapshot(
                chief_complaint="sudden chest pain and shortness of breath after a long flight",
                symptoms=[
                    Symptom(name="pleuritic chest pain", onset="today", severity="moderate"),
                    Symptom(name="shortness of breath"),
                    Symptom(name="calf pain and swelling"),
                ],
            ),
            TurnSnapshot(
                chief_complaint="sudden chest pain and shortness of breath after a long flight",
                symptoms=[
                    Symptom(name="pleuritic chest pain", onset="today", severity="moderate"),
                    Symptom(name="shortness of breath"),
                    Symptom(name="calf pain and swelling"),
                ],
                running_summary=(
                    "35-year-old on 12-hour flight yesterday, now with pleuritic chest pain and "
                    "dyspnea. Right calf swollen — DVT suspected. Wells score elevated."
                ),
            ),
        ],
    ),
    EvalCase(
        name="Pregnancy (missed period, nausea, pelvic pain)",
        target_exemplar_id="ex_pregnancy_001",
        turns=[
            TurnSnapshot(
                chief_complaint="missed period and nausea in a woman of reproductive age",
            ),
            TurnSnapshot(
                chief_complaint="missed period and nausea in a woman of reproductive age",
                symptoms=[
                    Symptom(name="missed period", onset="6 weeks ago"),
                    Symptom(name="nausea and vomiting"),
                    Symptom(name="right-sided pelvic pain"),
                    Symptom(name="light vaginal spotting"),
                ],
            ),
            TurnSnapshot(
                chief_complaint="missed period and nausea in a woman of reproductive age",
                symptoms=[
                    Symptom(name="missed period", onset="6 weeks ago"),
                    Symptom(name="nausea and vomiting"),
                    Symptom(name="right-sided pelvic pain"),
                    Symptom(name="light vaginal spotting"),
                ],
                running_summary=(
                    "28-year-old with 6-week amenorrhoea, positive home pregnancy test, "
                    "now with unilateral pelvic pain and spotting. Ectopic pregnancy must be excluded."
                ),
            ),
        ],
    ),
    EvalCase(
        name="Stimulant ACS (chest pain + cocaine use)",
        target_exemplar_id="ex_stimulant_acs_001",
        turns=[
            TurnSnapshot(
                chief_complaint="chest pain in a young man after recreational drug use",
            ),
            TurnSnapshot(
                chief_complaint="chest pain in a young man after recreational drug use",
                symptoms=[
                    Symptom(name="crushing chest pain", onset="1 hour ago", severity="severe"),
                    Symptom(name="diaphoresis"),
                    Symptom(name="palpitations"),
                ],
            ),
            TurnSnapshot(
                chief_complaint="chest pain in a young man after recreational drug use",
                symptoms=[
                    Symptom(name="crushing chest pain", onset="1 hour ago", severity="severe"),
                    Symptom(name="diaphoresis"),
                    Symptom(name="palpitations"),
                ],
                running_summary=(
                    "32-year-old male, cocaine use 2 hours ago, now with ACS-pattern chest pain. "
                    "Coronary vasospasm and type 2 MI on the differential alongside STEMI."
                ),
            ),
        ],
    ),
    EvalCase(
        name="SLE screen (joint pain, rash, young woman)",
        target_exemplar_id="ex_sle_screen_001",
        turns=[
            TurnSnapshot(
                chief_complaint="joint pain and facial rash in a young woman",
            ),
            TurnSnapshot(
                chief_complaint="joint pain and facial rash in a young woman",
                symptoms=[
                    Symptom(name="symmetric joint pain", onset="3 months"),
                    Symptom(name="butterfly rash across cheeks and nose"),
                    Symptom(name="fatigue"),
                    Symptom(name="hair loss"),
                ],
            ),
            TurnSnapshot(
                chief_complaint="joint pain and facial rash in a young woman",
                symptoms=[
                    Symptom(name="symmetric joint pain", onset="3 months"),
                    Symptom(name="butterfly rash across cheeks and nose"),
                    Symptom(name="fatigue"),
                    Symptom(name="hair loss"),
                ],
                running_summary=(
                    "24-year-old woman with 3 months of symmetric polyarthralgia, malar rash, "
                    "alopecia, and fatigue. Multi-system involvement pattern consistent with SLE."
                ),
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(snap: TurnSnapshot, idx: int) -> PatientProfile:
    return PatientProfile(
        session_id=f"eval-{idx}",
        demographics=Demographics(),
        chief_complaint=snap.chief_complaint,
        symptoms=snap.symptoms,
        history=History(),
        running_summary=snap.running_summary,
    )


def _hit_rate_random(
    pool,
    profile: PatientProfile,
    target_id: str,
    n_runs: int,
    k: int,
    rng: random.Random,
) -> float:
    hits = sum(
        1
        for _ in range(n_runs)
        if target_id in [ex.exemplar_id for ex in select_random(pool, n=k, rng=rng)]
    )
    return hits / n_runs


def _hit_mmr(profile: PatientProfile, target_id: str, k: int) -> float:
    result = get_exemplars_for_profile(profile, n=k)
    return 1.0 if target_id in [ex.exemplar_id for ex in result] else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_runs: int) -> None:
    pool = load_pool(_EXEMPLARS_DIR)
    rng = random.Random(42)

    col_w = 52
    turn_labels = ["Turn 1", "Turn 2", "Turn 3"]

    header = f"{'Case':<{col_w}} {'Strategy':<10} " + "  ".join(f"{t:>8}" for t in turn_labels)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    failures: list[str] = []

    for case in EVAL_CASES:
        print(f"\n{case.name}")
        print(f"  target: {case.target_exemplar_id}")

        rand_rates: list[Optional[float]] = []
        mmr_rates: list[Optional[float]] = []

        for turn_idx, snap in enumerate(case.turns):
            profile = _make_profile(snap, turn_idx)
            rand_rates.append(_hit_rate_random(pool, profile, case.target_exemplar_id, n_runs, TOP_K, rng))
            mmr_rates.append(_hit_mmr(profile, case.target_exemplar_id, TOP_K))

        def fmt(rates: list) -> str:
            return "  ".join(f"{r:>7.0%}" if r is not None else "     N/A" for r in rates)

        print(f"  {'Random (N=' + str(n_runs) + ')':<10}  {fmt(rand_rates)}")
        print(f"  {'MMR':<10}  {fmt(mmr_rates)}")

        if case.key_case:
            turn2_mmr = mmr_rates[1] if len(mmr_rates) > 1 else None
            if turn2_mmr is not None and turn2_mmr < PASS_THRESHOLD:
                failures.append(
                    f"FAIL  {case.name}: MMR turn-2 hit rate = {turn2_mmr:.0%} (need >{PASS_THRESHOLD:.0%})"
                )

    print("\n" + "=" * len(header))

    if failures:
        print("\nPASS CRITERION FAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\nAll key cases passed (MMR turn-2 hit rate >{PASS_THRESHOLD:.0%}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval random vs MMR retrieval quality.")
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="Runs per case for random baseline.")
    args = parser.parse_args()
    main(args.n)
