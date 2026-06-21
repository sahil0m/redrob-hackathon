"""
integrity.py — Honeypot and disqualifier detection.

The submission spec (Section 7) is explicit: the dataset hides ~80 honeypots with
"subtly impossible" profiles, they are forced to ground-truth tier 0, and a
honeypot rate > 10% in the top-100 is an automatic Stage-3 disqualification.

Strategy (calibrated against the full 100k pool, not assumed):
  * Two impossibility signatures fire on a tight, rare set and match the spec's
    own examples almost verbatim:
        - HIGH_PROFICIENCY_ZERO_DURATION : "'expert' proficiency in skills with
          0 years used"  -> 21 candidates in the pool, all clearly absurd
          (e.g. a Mobile Developer claiming 'expert' MLflow, Photoshop AND
          Content Writing, each duration_months=0).
        - JOB_LONGER_THAN_CAREER : a single role whose duration exceeds the
          candidate's entire claimed experience -> 21 candidates. This is the
          spec's "8 years at a company founded 3 years ago" family.
    These two sets do not overlap (verified) and together flag 42 profiles with
    essentially zero false positives. We treat them as a HARD GATE.
  * We deliberately do NOT widen the net to noisier signatures (e.g. a skill
    used a bit longer than the career), because at loose thresholds those fire on
    13k+ ordinary profiles and would throw away real candidates. The spec says a
    good ranker "will naturally avoid" the rest — and our role-fit scorer does,
    because impossible profiles also tend to be off-role.

We return structured `IntegrityReport`s, not just booleans, so the reasoning
layer can cite the exact reason and so the run prints an auditable summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from . import jd_spec
from .data import CandidateView, parse_date


@lru_cache(maxsize=None)
def _compiled_terms(terms: tuple[str, ...]) -> "re.Pattern":
    """Compile a term-set into ONE word-boundary alternation, cached per set.
    One scan per candidate instead of len(terms) scans — keeps the full-pool run
    well within budget."""
    alt = "|".join(re.escape(t.strip()) for t in terms if t.strip())
    return re.compile(r"\b(?:" + alt + r")\b")


def _count_word_terms(text: str, terms) -> int:
    """Count whole-word/phrase occurrences of `terms` in `text`.

    Plain substring matching is unsafe for short domain words — e.g. "search"
    is a substring of "research" and "text" of "context", which would silently
    corrupt the CV/speech-vs-NLP-IR comparison below. We anchor at word
    boundaries so "search" matches "search" but not "research", using a single
    precompiled alternation per term-set.
    """
    return len(_compiled_terms(tuple(terms)).findall(text))


@dataclass
class IntegrityReport:
    is_honeypot: bool = False
    honeypot_reasons: list[str] = field(default_factory=list)
    # Disqualifier severities are penalties in [0,1] subtracted/multiplied later,
    # not hard rejections (the JD hedges most disqualifiers with "probably").
    disqualifier_flags: dict[str, str] = field(default_factory=dict)


def detect_honeypot(c: CandidateView) -> tuple[bool, list[str]]:
    """Return (is_honeypot, reasons). Hard, high-precision impossibility checks."""
    reasons: list[str] = []
    yoe = c.years_of_experience

    # Signature 1: advanced/expert proficiency claimed with zero months of use.
    for s in c.skills:
        if s.get("proficiency") in ("advanced", "expert") and s.get("duration_months", 1) == 0:
            reasons.append(
                f"claims '{s.get('proficiency')}' in '{s.get('name')}' with 0 months of use"
            )
            break

    # Signature 2: a single role lasting longer than the entire claimed career.
    # (+6 months slack absorbs rounding in years_of_experience.)
    for j in c.career:
        dur = j.get("duration_months", 0) or 0
        if yoe > 0 and dur > yoe * 12 + 6:
            reasons.append(
                f"role at '{j.get('company')}' lasted {dur} months but total experience is "
                f"only {yoe:.1f} years"
            )
            break

    # Signature 3 (defensive, rarely fires): an end_date earlier than its start_date.
    for j in c.career:
        sd, ed = parse_date(j.get("start_date")), parse_date(j.get("end_date"))
        if sd and ed and ed < sd:
            reasons.append(
                f"role at '{j.get('company')}' ends ({ed}) before it starts ({sd})"
            )
            break

    return (len(reasons) > 0, reasons)


def _career_companies(c: CandidateView) -> list[str]:
    return [str(j.get("company", "")).lower() for j in c.career]


def detect_disqualifiers(c: CandidateView) -> dict[str, str]:
    """
    Return {flag_name: human_explanation} for each JD disqualifier the candidate
    trips. These feed a penalty in scoring, not an outright reject.
    """
    flags: dict[str, str] = {}
    title = c.current_title.lower()
    full_text = c.sparse_text  # already lowercased

    # --- Only-consulting career (JD penalizes ONLY if the WHOLE career is services,
    #     with an explicit carve-out for prior product-company experience). ---
    companies = _career_companies(c)
    if companies:
        def is_consulting(name: str) -> bool:
            return any(f in name for f in jd_spec.CONSULTING_FIRMS)

        if all(is_consulting(co) for co in companies if co):
            flags["only_consulting"] = (
                "entire career has been at IT-services/consulting firms "
                "(JD: explicit anti-pattern unless there is prior product-company experience)"
            )

    # --- Title-chaser: many short stints across distinct employers. ---
    durations = [j.get("duration_months", 0) or 0 for j in c.career]
    distinct_companies = len({co for co in companies if co})
    if len(c.career) >= jd_spec.TITLE_CHASER_MIN_JOBS and distinct_companies >= jd_spec.TITLE_CHASER_MIN_JOBS:
        srt = sorted(durations)
        mid = len(srt) // 2
        median = srt[mid] if len(srt) % 2 else (srt[mid - 1] + srt[mid]) / 2
        if median and median < jd_spec.TITLE_CHASER_MAX_MEDIAN_TENURE_MONTHS:
            flags["title_chaser"] = (
                f"median tenure ~{median:.0f} months across {distinct_companies} employers "
                "(JD: wants 3+ year commitment, not 1.5-year hops)"
            )

    # --- CV/speech/robotics dominant without NLP/IR exposure. ---
    # Word-boundary matching: avoids "search" matching "research", "text" "context".
    other_domain_hits = _count_word_terms(full_text, jd_spec.OTHER_DOMAIN_TERMS)
    nlp_ir_hits = _count_word_terms(full_text, jd_spec.NLP_IR_TERMS)
    if other_domain_hits >= 3 and nlp_ir_hits == 0:
        flags["wrong_ml_domain"] = (
            "profile is dominated by computer-vision/speech/robotics terms with no "
            "NLP or information-retrieval signal (JD: would be re-learning fundamentals)"
        )

    # --- Pure-keyword-stuffer: AI skills listed but title and career show no ML work.
    #     This is the JD's marquee trap ("all the AI keywords ... but title is
    #     Marketing Manager is not a fit"). ---
    ai_skill_terms = set(jd_spec.RETRIEVAL_TECH_TERMS + jd_spec.LLM_TERMS)
    ai_skill_ct = sum(
        1 for s in c.skills if any(t.strip() in str(s.get("name", "")).lower() for t in ai_skill_terms)
    )
    career_text = " ".join(str(j.get("description", "")) for j in c.career).lower()
    career_shows_ml = any(
        t in career_text
        for t in ("machine learning", "ml model", "embedding", "retrieval", "ranking",
                  "recommendation", "nlp", "neural", "llm", "model training")
    )
    title_is_technical = any(
        t in title for t in ("engineer", "developer", "scientist", "ml", "ai", "data", "research")
    )
    if ai_skill_ct >= 4 and not career_shows_ml and not title_is_technical:
        flags["keyword_stuffer"] = (
            f"lists {ai_skill_ct} AI/ML skills but the title is '{c.current_title}' and the "
            "career history shows no ML work (JD: keyword stuffing is an explicit trap)"
        )

    return flags


def assess(c: CandidateView) -> IntegrityReport:
    is_hp, hp_reasons = detect_honeypot(c)
    dq = detect_disqualifiers(c)
    return IntegrityReport(is_honeypot=is_hp, honeypot_reasons=hp_reasons, disqualifier_flags=dq)
