"""
Confidence plateau diagnostic script — Phase 6 item 4.

Tests three hypotheses for why confidence plateaus near 0.6:
  A — Prompt anchoring (the calibration scale anchors the model below 0.65)
  B — Model calibration ceiling (Llama-3.3-70b simply won't output > 0.65)
  C — Threshold too high (0.65 is appropriate; model is correctly uncertain)

Usage:
    python scripts/diagnose_confidence.py --hypothesis b
    python scripts/diagnose_confidence.py --hypothesis a
    python scripts/diagnose_confidence.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loop1.config import config
from loop1.llm import call_llm

MODEL = config["models"]["doctor"]

# ---------------------------------------------------------------------------
# Hypothesis B — model ceiling test
# One-shot: ask the model directly for confidence on a clear-cut, fully-confirmed case.
# If it stays <= 0.65 here, it's a model ceiling (Hypothesis B).
# ---------------------------------------------------------------------------

_HYP_B_SYSTEM = """You are an expert diagnostic physician. Output a single JSON object only.

{
  "confidence_to_stop": <float 0.0-1.0, your confidence that you have enough information to name a primary diagnosis>,
  "reasoning": "<brief explanation of your confidence level>"
}"""

_MEASLES_FULL = """
Patient: 4yo male, attends daycare.
Chief complaint: Fever and rash for 3 days.

Confirmed findings (all explicitly stated by parent):
- Fever 39.8°C, started 4 days ago
- Runny nose and cough started same day as fever
- White spots on inner cheeks (Koplik spots) confirmed 2 days ago, now fading
- Rash appeared on face 3 days ago, spreading downward to trunk and arms — classic cephalocaudal spread
- Three other children at daycare diagnosed with measles this week
- Child is NOT vaccinated (parents declined MMR)
- No travel outside local area
- No other diagnoses fit this constellation of findings

Given ALL of the above confirmed information, what is your confidence_to_stop for a primary diagnosis of measles?
"""

_CVST_FULL = """
Patient: 34yo female, graphic designer.
Chief complaint: Headache and dizziness for 5 days.

Confirmed findings:
- Severe progressive headache, worst-ever, started 5 days ago
- Headache is worse when lying down and improved slightly when upright
- Taking oral contraceptive pill (OCP) for 3 years
- Smokes 10 cigarettes/day
- Had a similar but milder headache episode 2 months ago that resolved
- Dizziness and blurred vision in right eye confirmed
- No fever, no neck stiffness
- MRI with venography would be needed to confirm, but clinical picture is compelling

Given ALL of the above confirmed information, what is your confidence_to_stop for a primary diagnosis of cerebral venous sinus thrombosis (CVST)?
"""

_ACS_FULL = """
Patient: 58yo male, accountant, BMI 31.
Chief complaint: Chest pain and shortness of breath for 2 hours.

Confirmed findings:
- Central crushing chest pain radiating to left arm, started 2 hours ago
- Pain is 8/10 in severity, not relieved by rest
- Associated diaphoresis (cold sweat), nausea
- Shortness of breath at rest
- History of hypertension and hypercholesterolaemia, on lisinopril and statin
- Father had MI at age 62
- Smoker, 20 pack-years, quit 5 years ago
- No recent travel, no leg swelling, no pleuritic component to pain

Given ALL of the above confirmed information, what is your confidence_to_stop for a primary diagnosis of STEMI/NSTEMI (acute coronary syndrome)?
"""


def test_hypothesis_b() -> None:
    print("\n" + "=" * 60)
    print("HYPOTHESIS B — Model calibration ceiling")
    print("Testing 3 fully-confirmed cases: measles, CVST, ACS")
    print("If model stays <= 0.65 on ALL three, it's a ceiling.")
    print("=" * 60)

    cases = [
        ("Measles (case_06)", _MEASLES_FULL),
        ("CVST (case_02)", _CVST_FULL),
        ("ACS (case_01)", _ACS_FULL),
    ]

    results: list[tuple[str, float, str]] = []

    for case_name, user_content in cases:
        print(f"\n  Testing {case_name}...")
        try:
            raw = call_llm(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _HYP_B_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=256,
            )
            import re
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
            data = json.loads(raw)
            confidence = data["confidence_to_stop"]
            reasoning = data.get("reasoning", "")
            results.append((case_name, confidence, reasoning))
            print(f"    confidence_to_stop = {confidence:.2f}")
            print(f"    reasoning: {reasoning[:120]}...")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            results.append((case_name, -1.0, str(exc)))

    print("\n  --- Summary ---")
    max_conf = max(r[1] for r in results if r[1] >= 0)
    for name, conf, _ in results:
        flag = " ← CEILING SUSPECTED" if conf <= 0.65 and conf >= 0 else ""
        print(f"  {name}: {conf:.2f}{flag}")

    print()
    if all(r[1] <= 0.65 for r in results if r[1] >= 0):
        print("  VERDICT: Hypothesis B CONFIRMED — model stays at or below 0.65 even on")
        print("  clear-cut cases with all features confirmed. This is a model calibration")
        print("  ceiling. Fix: lower threshold to 0.55 in config.yaml.")
    elif max_conf >= 0.75:
        print("  VERDICT: Hypothesis B REJECTED — model can reach > 0.65 when all evidence")
        print("  is confirmed. The plateau in live sessions is a prompt or task issue.")
        print("  Investigate Hypothesis A (prompt anchoring).")
    else:
        print(f"  VERDICT: Model reaches max {max_conf:.2f} in isolation. Partial ceiling.")
        print("  The prompt anchoring (Hypothesis A) may also be contributing.")


# ---------------------------------------------------------------------------
# Hypothesis A — prompt anchoring analysis (no API call needed)
# ---------------------------------------------------------------------------

def test_hypothesis_a() -> None:
    print("\n" + "=" * 60)
    print("HYPOTHESIS A — Prompt anchoring analysis")
    print("Checking the confidence calibration scale in doctor_v0_4.txt")
    print("=" * 60)

    prompt_path = Path(__file__).parent.parent / "prompts" / "doctor_v0_4.txt"
    prompt_text = prompt_path.read_text()

    # Find the confidence calibration section
    start = prompt_text.find("## Confidence calibration")
    end = prompt_text.find("## Communication style")
    section = prompt_text[start:end].strip() if start >= 0 else "(not found)"

    print("\n  Current calibration section:\n")
    for line in section.splitlines():
        print(f"    {line}")

    code_threshold = config["thresholds"]["confidence_to_stop"]
    print(f"\n  Code termination threshold: {code_threshold}")
    print(f"  Prompt should_stop rule:    >= 0.75")
    print()

    if code_threshold < 0.75:
        print(f"  *** MISMATCH DETECTED ***")
        print(f"  The code stops at {code_threshold} but the prompt tells the model")
        print(f"  to set should_stop=true only at >= 0.75.")
        print(f"  The calibration anchor '0.6–0.75: Strong leading diagnosis' teaches")
        print(f"  the model that 0.6 = 'still has questions to ask'. The model anchors")
        print(f"  to 0.6 because that's what the scale says to do at that confidence level.")
        print()
        print(f"  Candidate fix (Hypothesis A):")
        print(f"    Option A1 — Lower the scale threshold to match code:")
        print(f"      Change '0.8+: Ready to commit' → '0.65+: Ready to commit'")
        print(f"      Change 'should_stop >= 0.75' → 'should_stop >= {code_threshold}'")
        print(f"    Option A2 — Raise the code threshold to match prompt:")
        print(f"      Set confidence_to_stop: 0.75 in config.yaml")
        print(f"    Option A3 — Remove the calibration scale entirely,")
        print(f"      rely only on the decisive closure rule.")
    else:
        print(f"  No mismatch — code threshold matches prompt rule.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", choices=["a", "b", "all"], default="all")
    args = parser.parse_args()

    print("\nConfidence Plateau Diagnostic")
    print(f"Model: {MODEL}")
    print(f"Config threshold: {config['thresholds']['confidence_to_stop']}")

    if args.hypothesis in ("a", "all"):
        test_hypothesis_a()

    if args.hypothesis in ("b", "all"):
        test_hypothesis_b()

    print("\nDone.")


if __name__ == "__main__":
    main()
