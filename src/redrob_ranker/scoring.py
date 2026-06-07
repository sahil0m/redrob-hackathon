"""
scoring.py — Combine features, semantic similarity, penalties, behavioral
multiplier and the honeypot gate into a final fit score per candidate.

Score construction (all components in [0,1] unless noted):

    base   = Σ_w  weight[f] · feature[f]            # JD-justified linear blend
           + semantic_weight · semantic_sim          # hybrid-retrieval intent signal
    base   = base / (1 + semantic_weight)            # renormalise to [0,1]

    score  = base
           · behavioral_mult                         # availability (0.55..1.0)
           · disqualifier_penalty                    # JD anti-patterns (0..1)
    score  = 0.0  if honeypot                         # hard gate (forced tier-0)

The linear blend is a strong, fully-interpretable ranker on its own. The
XGBoost learning-to-rank model (ltr.py) is trained to refine the ordering using
the same components as input features; when present, its prediction REPLACES the
linear base before the multipliers, but we keep the linear score on every
candidate so we can always explain and fall back.

Disqualifier penalties are severities, not rejections, because the JD hedges most
disqualifiers ("we will *probably* not move forward"). The one near-hard signal
is the keyword-stuffer trap, which the JD calls out explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import jd_spec
from .data import CandidateView
from .features import FeatureBundle
from .integrity import IntegrityReport

# How much the hybrid-retrieval semantic similarity contributes relative to the
# summed feature weights (which total 1.0). 0.35 keeps structured features
# dominant while letting semantic intent meaningfully reorder.
SEMANTIC_WEIGHT = 0.35

# Disqualifier penalty multipliers (1.0 = no penalty). Each cites the JD.
DISQUALIFIER_PENALTIES = {
    # JD's marquee trap; the strongest signal a system "isn't reading profiles".
    "keyword_stuffer": 0.15,
    # "only worked at consulting firms ... in their entire career" — strong, but
    # the JD allows it if there was prior product experience (handled: this flag
    # only fires when the WHOLE career is services).
    "only_consulting": 0.55,
    # "title-chasers ... every 1.5 years ... we need 3+ years"
    "title_chaser": 0.6,
    # CV/speech/robotics without NLP/IR — "you'd be re-learning fundamentals"
    "wrong_ml_domain": 0.5,
}


@dataclass
class ScoredCandidate:
    candidate_id: str
    view: CandidateView
    features: FeatureBundle
    integrity: IntegrityReport
    linear_base: float = 0.0          # interpretable linear blend (pre-multipliers)
    base: float = 0.0                 # base used (linear or LTR), pre-multipliers
    disqualifier_penalty: float = 1.0
    score: float = 0.0                # final score after all multipliers + gate
    components: dict = field(default_factory=dict)  # for reasoning + audit


def _linear_base(fb: FeatureBundle) -> float:
    w = jd_spec.DEFAULT_WEIGHTS.as_dict()
    feat_sum = sum(w[name] * fb.values[name] for name in w)
    blended = feat_sum + SEMANTIC_WEIGHT * fb.semantic_sim
    return blended / (1.0 + SEMANTIC_WEIGHT)


def _disqualifier_penalty(report: IntegrityReport) -> float:
    penalty = 1.0
    for flag in report.disqualifier_flags:
        penalty *= DISQUALIFIER_PENALTIES.get(flag, 1.0)
    return penalty


def score_candidate(
    view: CandidateView,
    fb: FeatureBundle,
    report: IntegrityReport,
    ltr_base: float | None = None,
) -> ScoredCandidate:
    """
    Produce a ScoredCandidate. If ltr_base is provided (XGBoost prediction in
    [0,1]), it is used as the base; otherwise the interpretable linear blend is.
    The linear blend is always computed and stored for explanation/fallback.
    """
    linear = _linear_base(fb)
    base = ltr_base if ltr_base is not None else linear
    base = 0.0 if base < 0 else 1.0 if base > 1 else base

    penalty = _disqualifier_penalty(report)
    score = base * fb.behavioral_mult * penalty
    if report.is_honeypot:
        score = 0.0  # forced tier-0; never surface an impossible profile

    sc = ScoredCandidate(
        candidate_id=view.candidate_id,
        view=view,
        features=fb,
        integrity=report,
        linear_base=linear,
        base=base,
        disqualifier_penalty=penalty,
        score=score,
    )
    sc.components = {
        **fb.values,
        "semantic_sim": fb.semantic_sim,
        "behavioral_mult": fb.behavioral_mult,
        "disqualifier_penalty": penalty,
        "linear_base": linear,
        "base": base,
        "is_honeypot": report.is_honeypot,
        "disqualifiers": list(report.disqualifier_flags.keys()),
    }
    return sc
