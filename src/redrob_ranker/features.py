"""
features.py — Turn a CandidateView into a named, interpretable feature vector.

Every feature is a float in [0, 1] (higher = better fit) and is documented with
the JD rationale behind it. The same vector feeds two consumers:
  1. The transparent linear scorer (scoring.py), which blends a subset with the
     JD-justified weights in jd_spec.ScoringWeights.
  2. The XGBoost learning-to-rank model, which learns a non-linear blend over the
     full vector.

Keeping features named and bounded is what lets the reasoning layer say *which*
features earned or cost a candidate their rank, and lets us defend each number.

We compute against profile structure and free text, never against the
embedding similarity here — the semantic signal is injected separately in
scoring (so this module stays deterministic and unit-testable without a model).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from . import jd_spec
from .data import CandidateView, parse_date

_REF_DATE = dt.date.fromisoformat(jd_spec.REFERENCE_DATE)
_PROF_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def _contains_any(text: str, terms) -> int:
    """Count how many distinct terms appear in text (case-insensitive)."""
    t = text.lower()
    return sum(1 for term in terms if term in t)


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# --------------------------------------------------------------------------- #
# Individual feature computations
# --------------------------------------------------------------------------- #

def role_fit(c: CandidateView) -> float:
    """
    Does the candidate actually do this kind of work? Current title is the single
    most decisive anti-keyword-stuffer signal (JD: a Marketing Manager with all
    the AI skills is NOT a fit). We score the current title and the title history,
    weighting the current role most.
    """
    cur = c.current_title.lower()
    cur_core = any(t in cur for t in jd_spec.CORE_TITLE_TERMS)
    cur_adj = any(t in cur for t in jd_spec.ADJACENT_TITLE_TERMS)

    # Title history: has the candidate held a core/adjacent role before?
    hist_titles = " ".join(str(j.get("title", "")) for j in c.career).lower()
    hist_core = any(t in hist_titles for t in jd_spec.CORE_TITLE_TERMS)
    hist_adj = any(t in hist_titles for t in jd_spec.ADJACENT_TITLE_TERMS)

    score = 0.0
    if cur_core:
        score = 1.0
    elif cur_adj:
        score = 0.62              # plausible path in; JD blesses the data/backend route
    elif hist_core:
        score = 0.55              # was on-role recently, currently drifted
    elif hist_adj:
        score = 0.30
    else:
        score = 0.05              # off-role: HR, Accountant, Civil Eng, etc.

    # Small lift if the title history reinforces a current core/adjacent role.
    if cur_core and hist_core:
        score = 1.0
    if cur_adj and hist_core:
        score = min(1.0, score + 0.18)
    return _clip01(score)


def system_evidence(c: CandidateView) -> float:
    """
    Concrete evidence of building the systems the JD cares about — ranking,
    retrieval, recommendation, search — found in free-text career descriptions.
    This is how plain-language Tier-5s (who never write "RAG") get rewarded
    (JD: "if their career history shows they built a recommendation system at a
    product company, they're a fit").
    """
    career_text = " ".join(str(j.get("description", "")) for j in c.career)
    blob = f"{c.summary} {c.headline} {career_text}"
    strong = _contains_any(blob, jd_spec.SYSTEM_EVIDENCE_TERMS)       # ranking/retrieval/recsys: the bullseye
    support = _contains_any(blob, jd_spec.SUPPORTING_EVIDENCE_TERMS)  # production/ML/A-B: adjacent evidence

    # Strong evidence dominates (saturating); supporting evidence can lift a
    # plain-language candidate but is capped well below a true system-builder.
    strong_score = 1.0 - math.exp(-0.9 * strong)
    support_score = 0.55 * (1.0 - math.exp(-0.6 * support))
    return _clip01(max(strong_score, support_score) if strong else 0.7 * strong_score + support_score)


def must_have_skills(c: CandidateView) -> float:
    """
    The JD's "things you absolutely need": embeddings-retrieval tech, vector DBs /
    hybrid search, and evaluation frameworks for ranking. We look in BOTH the
    skills list and the career text, but weight career-text evidence higher
    (a skill *demonstrated in work* beats a skill merely *listed*). We also apply
    an endorsement/duration trust factor so lazy keyword-listing doesn't pay off.
    """
    skill_blob = " ".join(str(s.get("name", "")) for s in c.skills).lower()
    career_text = " ".join(str(j.get("description", "")) for j in c.career).lower()

    retr_skill = _contains_any(skill_blob, jd_spec.RETRIEVAL_TECH_TERMS)
    eval_skill = _contains_any(skill_blob, jd_spec.EVAL_FRAMEWORK_TERMS)
    retr_work = _contains_any(career_text, jd_spec.RETRIEVAL_TECH_TERMS)
    eval_work = _contains_any(career_text, jd_spec.EVAL_FRAMEWORK_TERMS)

    # Trust factor on listed skills: average (endorsements>0 OR duration>=12mo)
    # over the retrieval/eval-relevant listed skills. A skill with 0 endorsements
    # and 0 months used contributes little.
    relevant_terms = set(jd_spec.RETRIEVAL_TECH_TERMS + jd_spec.EVAL_FRAMEWORK_TERMS)
    trust_vals = []
    for s in c.skills:
        name = str(s.get("name", "")).lower()
        if any(rt.strip() in name for rt in relevant_terms):
            endorsed = (s.get("endorsements", 0) or 0) > 0
            seasoned = (s.get("duration_months", 0) or 0) >= 12
            trust_vals.append(1.0 if (endorsed or seasoned) else 0.35)
    trust = sum(trust_vals) / len(trust_vals) if trust_vals else 1.0

    listed = _clip01(1.0 - math.exp(-0.7 * (retr_skill + eval_skill))) * trust
    demonstrated = _clip01(1.0 - math.exp(-0.9 * (retr_work + eval_work)))
    # Demonstrated-in-work is worth more than merely-listed.
    return _clip01(0.45 * listed + 0.55 * demonstrated)


def experience_fit(c: CandidateView) -> float:
    """
    Experience in the JD's band, weighted toward product (not services) time.
    JD ideal: 6-8 yrs total, 4-5 in applied ML at product companies. The 5-9
    is a soft range, so we use a smooth trapezoid: full credit in [6,8], tapering
    to ACCEPTABLE bounds, never a hard cut.
    """
    yoe = c.years_of_experience
    lo, hi = jd_spec.IDEAL_YOE_LOW, jd_spec.IDEAL_YOE_HIGH
    alo, ahi = jd_spec.ACCEPTABLE_YOE_LOW, jd_spec.ACCEPTABLE_YOE_HIGH
    if lo <= yoe <= hi:
        band = 1.0
    elif yoe < lo:
        band = _clip01((yoe - alo) / (lo - alo)) if lo > alo else 0.0
    else:  # yoe > hi
        band = _clip01((ahi - yoe) / (ahi - hi)) if ahi > hi else 0.0

    # Product-vs-services: fraction of career months at non-consulting companies.
    total_m = sum(j.get("duration_months", 0) or 0 for j in c.career)
    product_m = 0
    for j in c.career:
        co = str(j.get("company", "")).lower()
        if not any(f in co for f in jd_spec.CONSULTING_FIRMS):
            product_m += j.get("duration_months", 0) or 0
    product_frac = (product_m / total_m) if total_m else 0.5

    # Blend: the band is the gate, product experience modulates it.
    return _clip01(band * (0.55 + 0.45 * product_frac))


def bonus_skills(c: CandidateView) -> float:
    """JD 'nice to have': LLM fine-tuning (LoRA/QLoRA/PEFT), learning-to-rank,
    HR-tech exposure, open-source. Lightly rewarded — never a primary driver."""
    blob = f"{c.summary} {' '.join(str(j.get('description','')) for j in c.career)} " \
           f"{' '.join(str(s.get('name','')) for s in c.skills)}"
    hits = _contains_any(blob, jd_spec.LLM_TERMS)
    return _clip01(1.0 - math.exp(-0.6 * hits))


def location_fit(c: CandidateView) -> float:
    """
    JD: Pune/Noida preferred; Hyderabad/Mumbai/Delhi-NCR/Bangalore welcome;
    willing-to-relocate counts; outside India is case-by-case (no visa sponsorship).
    """
    loc = c.location.lower()
    country = c.country.lower()
    relocate = bool(c.signals.get("willing_to_relocate", False))

    if any(city in loc for city in jd_spec.PREFERRED_CITIES):
        return 1.0
    if any(city in loc for city in jd_spec.TIER1_INDIAN_CITIES):
        return 0.85
    if "india" in country:
        return 0.7 if relocate else 0.55
    # Outside India: only meaningful if willing to relocate (still no visa help).
    return 0.35 if relocate else 0.15


def behavioral_multiplier(c: CandidateView) -> tuple[float, dict]:
    """
    A MULTIPLIER in [floor, 1.0] on the fit score (not an additive feature), so a
    great-but-quiet candidate still beats a mediocre-but-chatty one. JD: a
    perfect-on-paper candidate who hasn't logged in for 6 months with a 5%
    response rate is "not actually available — down-weight appropriately."

    Returns (multiplier, components) where components explains the contributions
    so the reasoning layer can cite them.
    """
    s = c.signals
    comp: dict[str, float] = {}

    # Recency of activity.
    last_active = parse_date(s.get("last_active_date"))
    if last_active:
        days = (_REF_DATE - last_active).days
        recency = _clip01(1.0 - max(0, days - 30) / 300.0)  # full credit <30d, ~0 by ~11mo
    else:
        recency = 0.5
    comp["recency"] = recency

    # Recruiter responsiveness.
    rr = s.get("recruiter_response_rate")
    responsiveness = _clip01(rr) if isinstance(rr, (int, float)) else 0.5
    comp["responsiveness"] = responsiveness

    # Open to work + verification (light trust signals).
    open_flag = 1.0 if s.get("open_to_work_flag") else 0.6
    comp["open_to_work"] = open_flag

    verified = sum(bool(s.get(k)) for k in ("verified_email", "verified_phone", "linkedin_connected"))
    verification = 0.8 + 0.2 * (verified / 3.0)  # 0.8..1.0
    comp["verification"] = verification

    # Interview reliability (attended scheduled interviews).
    icr = s.get("interview_completion_rate")
    reliability = _clip01(icr) if isinstance(icr, (int, float)) else 0.7
    comp["interview_reliability"] = reliability

    # Combine: weighted geometric-ish blend, then map to a gentle [0.55, 1.0] band
    # so behavior re-orders ties and nudges, but never dominates fit.
    raw = (0.42 * responsiveness + 0.28 * recency + 0.12 * open_flag
           + 0.10 * reliability + 0.08 * verification)
    multiplier = 0.55 + 0.45 * _clip01(raw)
    return multiplier, comp


# --------------------------------------------------------------------------- #
# Bundled feature vector
# --------------------------------------------------------------------------- #

FEATURE_NAMES = [
    "role_fit",
    "system_evidence",
    "must_have_skills",
    "experience_fit",
    "bonus_skills",
    "location_fit",
]


@dataclass
class FeatureBundle:
    values: dict[str, float]              # the named [0,1] features above
    behavioral_mult: float                # multiplier in [0.55, 1.0]
    behavioral_components: dict           # breakdown for reasoning
    semantic_sim: float = 0.0             # filled in by scoring (hybrid retrieval)

    def vector(self) -> list[float]:
        """Ordered numeric vector for the XGBoost model (features + behavior + semantic)."""
        return [self.values[n] for n in FEATURE_NAMES] + [self.behavioral_mult, self.semantic_sim]


def extract(c: CandidateView) -> FeatureBundle:
    vals = {
        "role_fit": role_fit(c),
        "system_evidence": system_evidence(c),
        "must_have_skills": must_have_skills(c),
        "experience_fit": experience_fit(c),
        "bonus_skills": bonus_skills(c),
        "location_fit": location_fit(c),
    }
    mult, comp = behavioral_multiplier(c)
    return FeatureBundle(values=vals, behavioral_mult=mult, behavioral_components=comp)


# Names of every column in FeatureBundle.vector(), for XGBoost feature importance.
VECTOR_NAMES = FEATURE_NAMES + ["behavioral_mult", "semantic_sim"]
