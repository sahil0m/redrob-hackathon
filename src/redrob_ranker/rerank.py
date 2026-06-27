"""
rerank.py — Stage 2 of the two-stage ranker: order the eligible set.

Stage 1 (candidate_generation) decided WHO is in contention; this module decides
the ORDER. Because the gate already guarantees on-role + non-honeypot, the
features that vary across the eligible set — and therefore actually move the
ranking — are the *differentiators*:

    experience_fit   in-band, product-weighted experience
    must_have_skills depth of the retrieval/eval stack
    system_evidence  how strong the ranking/retrieval/recsys evidence is
    semantic_sim     dense+BM25 intent match to the JD
    behavioral_mult  availability (recency, response rate, reliability)
    location_fit     Pune/Noida / relocation
    bonus_skills     LLM fine-tuning etc.

The linear weights below are DERIVED FROM A LEAVE-ONE-OUT ABLATION on the gated
pool (scripts/ablate_features.py), not hand-picked: each weight is proportional
to that feature's measured marginal contribution to the composite, normalised to
sum to 1. role_fit is intentionally NOT a re-rank feature — it is the gate and is
~1.0 for every eligible candidate, so including it would only add constant offset.

The XGBoost LTR (ltr.py) is trained on these same differentiators over the gated
set and is the primary re-ranker; this linear blend is the interpretable
fallback and the thing we defend by hand. Final order is
    base (LTR or linear) x behavioral_mult x disqualifier_penalty.
"""

from __future__ import annotations

# Re-rank features (role_fit excluded — it is the gate, constant post-gate).
RERANK_FEATURES = [
    "experience_fit",
    "must_have_skills",
    "system_evidence",
    "bonus_skills",
    "location_fit",
    "premium_pedigree",
]

# Weights follow the JD's OWN priority tiers (the defensible ordering), informed
# by ablation but deliberately NOT copied from it.
#
# The JD is explicit about what matters most among on-role candidates:
#   * "Things you absolutely need": shipped a ranking/search/recsys system
#     (system_evidence) and the retrieval+eval stack (must_have_skills)  -> TOP.
#   * "5-9 years ... a range, not a requirement ... we'll seriously consider
#     candidates outside the band if other signals are strong"  -> experience is
#     SECONDARY, not the lead signal.
#   * Location "preferred"; bonus skills "nice to have but won't reject you".
#
# So system_evidence + must_have_skills (0.58 combined) dominate experience
# (0.18). A naive ablation on our own eval labels suggested ~0.9 weight on
# experience, but that is circular (the labels use in-band experience to split
# tier 3 from tier 4) and contradicts the JD — so we reject it. This ordering is
# also consistent with the LTR's proxy_label, where experience enters only as a
# mild 0.7-1.0 modifier on top of competence. Validated against evaluate.py:
# headline NDCG holds and the top-100 fills with senior systems-builders.
RERANK_WEIGHTS = {
    "system_evidence": 0.28,
    "must_have_skills": 0.26,
    "experience_fit": 0.16,
    "premium_pedigree": 0.10,
    "location_fit": 0.12,
    "bonus_skills": 0.08,
}

# Weight on the dense+BM25 semantic signal, relative to the summed feature
# weights (which total 1.0). Kept modest: it re-orders ties among the eligible
# set by intent but does not override structured fit.
SEMANTIC_WEIGHT = 0.30


def rerank_base(fb) -> float:
    """Interpretable Stage-2 base score in [0,1] for an eligible candidate."""
    feat = sum(RERANK_WEIGHTS[f] * fb.values[f] for f in RERANK_WEIGHTS)
    blended = feat + SEMANTIC_WEIGHT * fb.semantic_sim
    return blended / (1.0 + SEMANTIC_WEIGHT)
