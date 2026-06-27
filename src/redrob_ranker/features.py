"""
features.py — Turn a CandidateView into a named, interpretable feature vector.

Every feature is a float in [0, 1] (higher = better fit) and is documented with
the JD rationale behind it. The features feed both stages:
  1. role_fit drives the Stage-1 eligibility gate (candidate_generation.py).
  2. The differentiator features feed the Stage-2 re-ranker — both the
     interpretable linear blend (rerank.py) and the XGBoost LTR (ltr.py), which
     learns a non-linear blend over them.

Keeping features named and bounded is what lets the reasoning layer say *which*
features earned or cost a candidate their rank, and lets us defend each number.

We compute against profile structure and free text, never against the
embedding similarity here — the semantic signal is injected separately in
scoring (so this module stays deterministic and unit-testable without a model).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass

TIER_1_RE = re.compile(r'\b(iit|nit|bits pilani|iiit|dtu|nsut|iim|xlri)\b', re.IGNORECASE)
UNICORN_RE = re.compile(r'\b(flipkart|razorpay|cred|swiggy|zomato|ola|paytm|meesho|zepto|blinkit)\b', re.IGNORECASE)

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

    if cur_core:
        score = 1.0
    elif cur_adj:
        # adjacent title (data/backend/SWE) — plausible path in; lifted if the
        # candidate has also held a core role before (JD blesses this route).
        score = 0.80 if hist_core else 0.62
    elif hist_core:
        score = 0.55              # was on-role recently, currently drifted
    elif hist_adj:
        score = 0.30
    else:
        score = 0.05              # off-role: HR, Accountant, Civil Eng, etc.
    return _clip01(score)


def system_evidence(c: CandidateView) -> float:
    """
    Concrete evidence of building the systems the JD cares about — ranking,
    retrieval, recommendation, search — found in free-text career descriptions.
    This is how plain-language Tier-5s (who never write "RAG") get rewarded
    (JD: "if their career history shows they built a recommendation system at a
    product company, they're a fit").
    """
    strong_total = 0.0
    support_total = 0.0

    # Base profile text (no decay for summary/headline)
    base_blob = f"{c.summary} {c.headline}"
    strong_total += _contains_any(base_blob, jd_spec.SYSTEM_EVIDENCE_TERMS)
    support_total += _contains_any(base_blob, jd_spec.SUPPORTING_EVIDENCE_TERMS)

    # Explicitly sort career history before time decay
    sorted_roles = sorted(
        c.career,
        key=lambda r: (1 if r.get("is_current") else 0, str(r.get("start_date", ""))),
        reverse=True
    )

    for index, role in enumerate(sorted_roles):
        decay_factor = math.exp(-0.4 * index)
        desc = str(role.get("description", ""))
        strong_total += decay_factor * _contains_any(desc, jd_spec.SYSTEM_EVIDENCE_TERMS)
        support_total += decay_factor * _contains_any(desc, jd_spec.SUPPORTING_EVIDENCE_TERMS)

    # Strong evidence dominates (saturating); supporting evidence can lift a
    # plain-language candidate but is capped well below a true system-builder.
    strong_score = 1.0 - math.exp(-0.9 * strong_total)
    support_score = 0.55 * (1.0 - math.exp(-0.6 * support_total))
    return _clip01(max(strong_score, support_score) if strong_total else 0.7 * strong_score + support_score)


def must_have_skills(c: CandidateView) -> float:
    """
    The JD's "things you absolutely need": embeddings-retrieval tech, vector DBs /
    hybrid search, and evaluation frameworks for ranking. We look in BOTH the
    skills list and the career text, but weight career-text evidence higher
    (a skill *demonstrated in work* beats a skill merely *listed*). We also apply
    an endorsement/duration trust factor so lazy keyword-listing doesn't pay off.
    """
    skill_blob = " ".join(str(s.get("name", "")) for s in c.skills).lower()
    retr_skill = _contains_any(skill_blob, jd_spec.RETRIEVAL_TECH_TERMS)
    eval_skill = _contains_any(skill_blob, jd_spec.EVAL_FRAMEWORK_TERMS)

    retr_work_total = 0.0
    eval_work_total = 0.0

    # Explicitly sort career history before time decay
    sorted_roles = sorted(
        c.career,
        key=lambda r: (1 if r.get("is_current") else 0, str(r.get("start_date", ""))),
        reverse=True
    )

    for index, role in enumerate(sorted_roles):
        decay_factor = math.exp(-0.4 * index)
        desc = str(role.get("description", "")).lower()
        retr_work_total += decay_factor * _contains_any(desc, jd_spec.RETRIEVAL_TECH_TERMS)
        eval_work_total += decay_factor * _contains_any(desc, jd_spec.EVAL_FRAMEWORK_TERMS)

    # Trust factor on listed skills: average (endorsements>0 OR duration>=12mo)
    # over the retrieval/eval-relevant listed skills. A skill with 0 endorsements
    # and 0 months used contributes little.
    relevant_terms = set(jd_spec.RETRIEVAL_TECH_TERMS + jd_spec.EVAL_FRAMEWORK_TERMS)
    trust_vals = []
    for s in c.skills:
        name = str(s.get("name", "")).lower()
        if any(rt.strip() in name for rt in relevant_terms):
            if s.get("is_inferred"):
                trust_vals.append(0.20)
                continue
            endorsed = (s.get("endorsements", 0) or 0) > 0
            seasoned = (s.get("duration_months", 0) or 0) >= 12
            trust_vals.append(1.0 if (endorsed or seasoned) else 0.35)
    trust = sum(trust_vals) / len(trust_vals) if trust_vals else 1.0

    listed = _clip01(1.0 - math.exp(-0.7 * (retr_skill + eval_skill))) * trust
    demonstrated = _clip01(1.0 - math.exp(-0.9 * (retr_work_total + eval_work_total)))
    # Demonstrated-in-work is worth more than merely-listed.
    base = _clip01(0.45 * listed + 0.55 * demonstrated)

    # Objective validation via Redrob skill-assessment scores (0-100). This is the
    # strongest anti-keyword-stuffer signal in the data: a candidate who *lists*
    # AI/ML skills but scored poorly on the platform's assessment of them is making
    # hollow claims; one who scored well is validated. Applied as a gentle
    # multiplier, neutral when the candidate took no relevant assessment (absence
    # is not a negative). JD/challenge: rank on demonstrated ability, not keywords.
    return _clip01(base * _assessment_factor(c))


# Skills whose assessment scores meaningfully validate technical credibility for
# this role (retrieval/eval stack + the broader AI/ML/NLP space).
_ASSESSMENT_RELEVANT = set(jd_spec.RETRIEVAL_TECH_TERMS + jd_spec.EVAL_FRAMEWORK_TERMS
                           + jd_spec.LLM_TERMS) | {
    "nlp", "machine learning", "deep learning", "ml", "recommendation", "ranking",
    "data science", "neural", "computer vision", "model", "pytorch", "tensorflow",
}


def _assessment_factor(c: CandidateView) -> float:
    """Multiplier in [~0.81, ~1.05] from Redrob skill_assessment_scores on the
    AI/ML-relevant skills. 1.0 (neutral) when no relevant assessment was taken."""
    scores = c.signals.get("skill_assessment_scores") or {}
    if not scores:
        return 1.0
    relevant = [
        v for k, v in scores.items()
        if isinstance(v, (int, float)) and any(t.strip() in str(k).lower() for t in _ASSESSMENT_RELEVANT)
    ]
    if not relevant:
        return 1.0
    avg = sum(relevant) / len(relevant) / 100.0     # normalise 0-100 -> 0-1
    # high assessment (≈1.0) -> 1.05 boost; low (≈0.25) -> ~0.81 reduction.
    return 0.75 + 0.30 * avg


def experience_fit(c: CandidateView) -> float:
    """
    Experience credit, weighted toward product (not services) time.

    The JD gives two numbers: the IDEAL "6-8 years" and the STATED requirement
    "5-9 years", and is explicit that "this is a range, not a requirement ...
    some people hit senior judgment at 4 years ... we'll seriously consider
    candidates outside the band if other signals are strong." So we do NOT punish
    the stated range: full credit across the ideal [6,8], still-strong credit
    (>=0.8) across the stated [5,9], and a gentle taper outside — never a cliff.
    Implemented as linear interpolation over JD-anchored points.
    """
    yoe = c.years_of_experience
    # (years, credit) anchors: ideal 6-8 = 1.0; stated 5 and 9 = 0.8; taper to 0.
    anchors = [(0.0, 0.0), (4.0, 0.45), (5.0, 0.80), (6.0, 1.0),
               (8.0, 1.0), (9.0, 0.80), (11.0, 0.40), (13.0, 0.0), (50.0, 0.0)]
    band = 0.0
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= yoe <= x1:
            band = y0 + (y1 - y0) * ((yoe - x0) / (x1 - x0)) if x1 > x0 else y0
            break

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
    HR-tech exposure, AND open-source / external validation. Lightly rewarded —
    never a primary driver.

    The JD explicitly values open-source ("Open-source contributions in the AI/ML
    space") and treats "closed-source proprietary systems for 5+ years without
    external validation" as an anti-pattern. The Redrob github_activity_score
    (0-100, or -1 if no GitHub) is the objective proxy for that, so a strong
    GitHub presence lifts this bonus. -1/absent is neutral (50% have no GitHub
    linked — absence is not penalised, since it's a 'nice to have')."""
    blob = f"{c.summary} {' '.join(str(j.get('description','')) for j in c.career)} " \
           f"{' '.join(str(s.get('name','')) for s in c.skills)}"
    hits = _contains_any(blob, jd_spec.LLM_TERMS)
    llm_bonus = 1.0 - math.exp(-0.6 * hits)

    gh = c.signals.get("github_activity_score")
    gh_bonus = _clip01(gh / 70.0) if isinstance(gh, (int, float)) and gh > 0 else 0.0

    # Either signal can carry the bonus; take the stronger so neither is required.
    return _clip01(max(llm_bonus, gh_bonus))


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

    # Notice period (JD: "We'd love sub-30-day notice ... 30+ day notice
    # candidates are still in scope but the bar gets higher"). Shorter = more
    # readily hireable. Full credit at <=30 days, gentle decline beyond.
    notice = s.get("notice_period_days")
    if isinstance(notice, (int, float)):
        notice_credit = _clip01(1.0 - max(0.0, notice - 30) / 200.0)  # 30→1.0, 90→0.7, 150→0.4
    else:
        notice_credit = 0.7
    comp["notice_period"] = notice_credit

    # Combine: weighted blend, then map to a gentle [0.55, 1.0] band so behavior
    # re-orders ties and nudges, but never dominates fit (the JD says behavior is
    # "a multiplier or modifier on top of skill-match scoring").
    raw = (0.36 * responsiveness + 0.26 * recency + 0.11 * open_flag
           + 0.09 * reliability + 0.08 * notice_credit + 0.10 * verification)
    multiplier = 0.55 + 0.45 * _clip01(raw)
    return multiplier, comp


# --------------------------------------------------------------------------- #
# Bundled feature vector
# --------------------------------------------------------------------------- #

def premium_pedigree(c: CandidateView) -> float:
    """
    Indian Industry Context: Tier-1 / Unicorn Signal.
    Deterministic local industry weightings. Looks for Tier-1 institutions
    and premium Indian Unicorns in education and career history.
    """
    score = 0.0
    # Check Education
    for edu in c.education:
        school = str(edu.get('school', '')).lower()
        if TIER_1_RE.search(school):
            score += 0.5
            break
            
    # Check Career History for Unicorns
    for role in c.career:
        company = str(role.get('company', '')).lower()
        if UNICORN_RE.search(company):
            score += 0.5
            if role.get('is_current', False):
                score += 0.25
            break
            
    return _clip01(score)

FEATURE_NAMES = [
    "role_fit",
    "system_evidence",
    "must_have_skills",
    "experience_fit",
    "bonus_skills",
    "location_fit",
    "premium_pedigree",
]


@dataclass
class FeatureBundle:
    values: dict[str, float]              # the named [0,1] features above
    behavioral_mult: float                # multiplier in [0.55, 1.0]
    behavioral_components: dict           # breakdown for reasoning
    semantic_sim: float = 0.0             # filled in by scoring (hybrid retrieval)

    def rerank_vector(self) -> list[float]:
        """Stage-2 re-rank vector for XGBoost: the differentiator features +
        semantic + behavioral. role_fit is excluded (it's the Stage-1 gate and is
        ≈1.0 across the eligible set, so it carries no ordering information)."""
        from .rerank import RERANK_FEATURES
        return [self.values[n] for n in RERANK_FEATURES] + [self.semantic_sim, self.behavioral_mult]


# role_fit is computed but used as the Stage-1 gate (candidate_generation), not
# as a Stage-2 re-rank weight. See rerank.RERANK_FEATURES for the re-rank set.


def extract(c: CandidateView) -> FeatureBundle:
    vals = {
        "role_fit": role_fit(c),
        "system_evidence": system_evidence(c),
        "must_have_skills": must_have_skills(c),
        "experience_fit": experience_fit(c),
        "bonus_skills": bonus_skills(c),
        "location_fit": location_fit(c),
        "premium_pedigree": premium_pedigree(c),
    }
    mult, comp = behavioral_multiplier(c)
    return FeatureBundle(values=vals, behavioral_mult=mult, behavioral_components=comp)
