"""
eval_labels.py — Independent synthetic relevance labels for offline evaluation.

We have no access to the hidden ground truth, so to MEASURE ranking quality
(instead of guessing) we build a principled proxy relevance grade for each
candidate and score our ranker against it with NDCG@10/50, MAP and P@10 — the
exact metrics the competition uses.

Crucial design point: these labels are derived from a DIFFERENT reasoning path
than the ranker's scoring features, so the evaluation is not circular. The
ranker scores via smooth, learned-blend features; these labels apply discrete,
rule-based JD judgements a human reviewer would make:

  Tier 0  honeypot / impossible profile, OR a hard JD disqualifier with no
          redeeming on-role evidence (keyword stuffer, consulting-only, etc.)
  Tier 1  off-role (wrong job family) — not relevant to a Senior AI Engineer hire
  Tier 2  adjacent role (data/backend/SWE) with little ML-systems evidence
  Tier 3  on-role OR strong adjacent with real ML/production evidence; "relevant"
  Tier 4  bullseye: on-role AI/ML title + ranking/retrieval/recsys evidence +
          experience in band + reachable

The grade is intentionally coarse and discrete; it encodes the JD's stated
"ideal candidate" and disqualifiers literally. It is a stand-in for human
judgement, NOT the truth — but it lets us compare design variants quantitatively
and catch regressions, which is what a real team needs.
"""

from __future__ import annotations

import datetime as dt

from . import jd_spec
from .data import CandidateView, parse_date
from .integrity import detect_disqualifiers, detect_honeypot

_REF = dt.date.fromisoformat(jd_spec.REFERENCE_DATE)

# Discrete title taxonomy (independent of features.role_fit's term lists by intent:
# here we make hard family decisions a recruiter would make).
_CORE_AI_TITLES = (
    "machine learning engineer", "ml engineer", "applied ml", "applied scientist",
    "ai engineer", "ai research engineer", "research engineer", "data scientist",
    "nlp engineer", "search engineer", "ranking engineer", "relevance engineer",
    "recommendation",
)
_ADJACENT_TITLES = (
    "data engineer", "analytics engineer", "backend engineer", "software engineer",
    "full stack", "platform engineer", "mlops", "machine learning",
)
_OFFROLE_FAMILIES = (
    "hr ", "human resource", "accountant", "civil engineer", "mechanical engineer",
    "graphic designer", "content writer", "sales", "marketing", "operations manager",
    "customer support", "project manager", "business analyst", "qa ", "frontend",
    "mobile developer", ".net", "java developer",
)


def _has_ml_systems_evidence(c: CandidateView) -> bool:
    text = (c.summary + " " + " ".join(str(j.get("description", "")) for j in c.career)).lower()
    return any(t in text for t in jd_spec.SYSTEM_EVIDENCE_TERMS)


def _has_production_ml(c: CandidateView) -> bool:
    text = (c.summary + " " + " ".join(str(j.get("description", "")) for j in c.career)).lower()
    return any(t in text for t in jd_spec.SUPPORTING_EVIDENCE_TERMS)


def _in_band(c: CandidateView) -> bool:
    return jd_spec.ACCEPTABLE_YOE_LOW <= c.years_of_experience <= jd_spec.ACCEPTABLE_YOE_HIGH


def _reachable(c: CandidateView) -> bool:
    s = c.signals
    la = parse_date(s.get("last_active_date"))
    recent = la is not None and (_REF - la).days <= jd_spec.STALE_DAYS
    rr = s.get("recruiter_response_rate")
    responsive = isinstance(rr, (int, float)) and rr >= 0.15
    return recent and responsive


def relevance_tier(c: CandidateView) -> int:
    """Return a 0-4 relevance grade from discrete JD rules (independent of the ranker)."""
    # Tier 0: impossible profile.
    is_hp, _ = detect_honeypot(c)
    if is_hp:
        return 0

    title = c.current_title.lower()
    is_core = any(t in title for t in _CORE_AI_TITLES)
    is_adjacent = any(t in title for t in _ADJACENT_TITLES)
    is_offrole = any(t in title for t in _OFFROLE_FAMILIES) and not is_core and not is_adjacent

    dq = detect_disqualifiers(c)
    # Keyword stuffer with off-role title => tier 0 (the marquee trap).
    if "keyword_stuffer" in dq:
        return 0

    sys_ev = _has_ml_systems_evidence(c)
    prod_ml = _has_production_ml(c)

    # Off-role and no ML evidence => not relevant.
    if is_offrole and not sys_ev and not prod_ml:
        return 0 if dq else 1

    # Bullseye: on-role AI title + ranking/retrieval/recsys evidence + in band + reachable.
    if is_core and sys_ev and _in_band(c) and _reachable(c) and not dq:
        return 4

    # Strong: on-role (or adjacent with systems evidence), real ML evidence.
    if (is_core or (is_adjacent and sys_ev)) and (sys_ev or prod_ml):
        # heavy disqualifier knocks a strong candidate down
        if dq:
            return 2
        return 3

    # Adjacent with only light ML evidence.
    if is_adjacent and (sys_ev or prod_ml):
        return 2 if not dq else 1

    # On-role title but thin evidence.
    if is_core:
        return 2 if not dq else 1

    # Everything else: weakly relevant at best.
    return 1
