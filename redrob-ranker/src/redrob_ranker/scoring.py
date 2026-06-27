"""
scoring.py — Final score per candidate, in the two-stage framing.

Stage 1 (candidate_generation) decides eligibility; this module computes the
Stage-2 score for the candidates that passed, and forces everyone else to 0.

Score construction (components in [0,1] unless noted):

    base   = rerank_base(features)                   # Stage-2 differentiator blend
                                                       # (or the XGBoost LTR prediction)
    score  = base
           · behavioral_mult                          # availability (0.55..1.0)
           · disqualifier_penalty                     # JD anti-patterns (0..1)
    score  = 0.0  if honeypot OR not eligible          # gated out

The re-rank base (rerank.py) excludes role_fit — that is the Stage-1 gate and is
≈1.0 for every eligible candidate, so it carries no ordering information. The
XGBoost LTR (ltr.py) refines the ordering over the same differentiators; the
interpretable linear blend is always computed and stored as the fallback we can
defend by hand.

Disqualifier penalties are severities, not rejections, because the JD hedges most
disqualifiers ("we will *probably* not move forward"). The one near-hard signal
is the keyword-stuffer trap, which the JD calls out explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import CandidateView
from .features import FeatureBundle
from .integrity import IntegrityReport

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
    """Stage-2 interpretable re-rank base. role_fit is excluded here on purpose:
    it is the Stage-1 gate (≈1.0 for every eligible candidate), so it adds only a
    constant offset to the order. See rerank.py for the weight rationale."""
    from .rerank import rerank_base
    return rerank_base(fb)


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
    eligible: bool = True,
) -> ScoredCandidate:
    """
    Produce a ScoredCandidate. If ltr_base is provided (XGBoost prediction in
    [0,1]), it is used as the Stage-2 base; otherwise the interpretable linear
    re-rank blend is. The linear blend is always computed and stored for
    explanation/fallback.

    `eligible` is the Stage-1 gate decision. Ineligible candidates (off-role,
    insufficient evidence) and honeypots are forced to score 0 so they can never
    surface in the top-100 — this is the candidate-generation half of the
    two-stage design.
    """
    linear = _linear_base(fb)
    base = ltr_base if ltr_base is not None else linear
    base = 0.0 if base < 0 else 1.0 if base > 1 else base

    penalty = _disqualifier_penalty(report)
    score = base * fb.behavioral_mult * penalty
    if report.is_honeypot or not eligible:
        score = 0.0  # gated out: impossible profile or off-role / thin evidence

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
