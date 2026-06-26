# Phase 7 Findings Report

Generated: 2026-06-09T10:40:19.181196+00:00

---

## 1. Diagnostic Accuracy

### In Pool (n=3)
- Top-1 correct: 1/3 (33%)
- Ground truth in top-3: 1/3 (33%)

### Per-disease breakdown

| Disease | Stratum | Top-1 Correct | In Top-3 | Turns | Confidence |
|---------|---------|--------------|----------|-------|------------|
| Pulmonary embolism | in_pool | ✓ | ✓ | 9 | 0.60 |
| Pericarditis | in_pool | ✗ | ✗ | — | 0.00 |
| Possible NSTEMI / STEMI | in_pool | ✗ | ✗ | 12 | 0.40 |
| SLE | in_pool | ✗ | ✗ | — | 0.00 |
| Unstable angina | in_pool | ✗ | ✗ | 12 | 0.40 |

---

## 2. Question Quality (Critic)

- Total turns critiqued: 33
- Mean question quality: 0.703
- Median question quality: 0.7
- Mean differential quality: 0.582
- Mean reasoning quality: 0.815

### Confidence calibration distribution

- well-calibrated: 7
- underconfident: 26

### Weakness category counts

- poor_differential: 17
- missed_red_flag: 8
- redundant_question: 4

---

## 3. Cross-Reference: Low Question Quality + Wrong Diagnosis

Sessions where question quality was poor AND the diagnosis was wrong:

| Patient | Mean Q Quality | Low-Quality Turns | Correct? | Weakness Categories |
|---------|---------------|------------------|----------|---------------------|

---

## 4. Categorized Failure Modes

- Wrong diagnosis, no matching exemplar: 1
- Wrong diagnosis, exemplar existed: 1
- Redundant questions (total turns): 4
- Missed red flags (total turns): 8
- Poor differential ordering (total turns): 17

---

## 5. Top Weakness Texts

- (1×) The doctor's question is reasonable but may not be the highest-yield option given the patient's symptoms.
- (1×) The differential ranking does not accurately reflect the confirmed evidence.
- (1×) The differential ranking does not accurately reflect the confirmed evidence, with Anxiety-Related Chest Pain ranked higher than Pulmonary Embolism despite the patient's symptoms of sharp chest pain and significant shortness of breath.
- (1×) The differential ranking could be improved to reflect the confirmed evidence more accurately.
- (1×) The doctor's question is reasonable but does not directly address the most critical aspect of the patient's symptoms and risk factors.

---

## 6. Phase 8 Recommendations

- **Prompt iteration needed**: top-1 accuracy below 50% — doctor prompt is underperforming.
- **Targeted exemplars for red-flag screening**: 8 turns flagged for missed red flags.
