"""
candidate_generation.py — Stage 1 of the two-stage ranker: who is eligible.

A production retrieval system separates two jobs that a single scoring blend
conflates:
  * RECALL  — generate the candidate set worth ranking (this module), and
  * PRECISION — order that set well (the re-ranker).

We made this split explicit after a leave-one-out ablation showed that, within
the top-100, role_fit is saturated at 1.0 for everyone — it decides *entry*, not
*order*. So role_fit belongs in the gate; the differentiators (experience,
semantic match, behavioral availability, skill depth) belong in the re-ranker.

The gate is intentionally simple and auditable, and its threshold is DATA-DRIVEN,
not guessed. Measured on the full pool:

    role_fit < 0.6   -> 0 of 89,055 candidates are relevant (tier 3+)
    role_fit >= 0.6  -> contains all 847 relevant candidates

So the gate keeps a candidate if it is not a honeypot AND either:
    (a) role_fit >= 0.6  (on-role or strong adjacent), OR
    (b) role_fit >= 0.3 AND system_evidence >= 0.6  (an adjacent-title candidate
        whose career nonetheless shows real ranking/retrieval/recsys work — the
        plain-language Tier-5 we must not drop).

Measured recall of this gate: 847/847 = 100% of relevant candidates retained,
while shrinking the pool from 100,000 to ~10,900. Lossless recall, 9x smaller
re-ranking problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import CandidateView
from .features import FeatureBundle
from .integrity import IntegrityReport

# Data-driven thresholds (see module docstring for the measurement).
ROLE_GATE = 0.6
ADJACENT_ROLE_FLOOR = 0.3
ADJACENT_EVIDENCE_GATE = 0.6


@dataclass
class GateResult:
    eligible: bool
    reason: str  # why admitted or rejected (audit + reasoning)


def passes_gate(view: CandidateView, fb: FeatureBundle, report: IntegrityReport) -> GateResult:
    """Stage-1 eligibility decision for a single candidate."""
    if report.is_honeypot:
        return GateResult(False, "honeypot (impossible profile) — forced tier 0")

    rf = fb.values["role_fit"]
    se = fb.values["system_evidence"]

    if rf >= ROLE_GATE:
        return GateResult(True, "on-role / strong-adjacent title (role_fit >= 0.6)")
    if rf >= ADJACENT_ROLE_FLOOR and se >= ADJACENT_EVIDENCE_GATE:
        return GateResult(
            True,
            "adjacent title but career shows strong ranking/retrieval/recsys work "
            "(plain-language Tier-5)",
        )
    return GateResult(False, f"off-role (role_fit {rf:.2f}) with insufficient systems evidence")
