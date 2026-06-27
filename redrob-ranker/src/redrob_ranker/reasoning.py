"""
reasoning.py — Generate the per-candidate `reasoning` string for the CSV.

Stage 4 samples 10 rows and checks reasoning for: specific profile facts, a
connection to JD requirements, honest acknowledgment of concerns, NO
hallucination, variation between rows, and tone consistent with the rank.

To pass all six, every sentence here is assembled from facts that actually drove
the candidate's score — pulled from the CandidateView and the feature
components. We never name a skill, employer, or experience that isn't in the
profile (anti-hallucination), and the praise/concern balance is keyed to the
candidate's own feature values (so tone tracks rank automatically). The phrasing
is varied by selecting from fact-specific clauses rather than a fixed template.

The output is plain, specific, and honest — exactly what the spec says ranks
highly ("Don't try to be impressive; try to be specific and honest.").
"""

from __future__ import annotations

from . import jd_spec
from .scoring import ScoredCandidate


def _years(c) -> str:
    return f"{c.view.years_of_experience:.1f} yrs"


def _strengths(sc: ScoredCandidate) -> list[str]:
    """Fact-grounded positive clauses, only emitted when the feature supports them."""
    v = sc.features.values
    view = sc.view
    out: list[str] = []

    # Role / title — cite the actual title.
    if v["role_fit"] >= 0.6:
        out.append(f"on-role as {view.current_title}")
    elif v["role_fit"] >= 0.3:
        out.append(f"adjacent background ({view.current_title}) with a plausible path into the role")

    # System evidence — only claim it if present; quote the kind of work found.
    if v["system_evidence"] >= 0.6:
        kinds = [t for t in jd_spec.SYSTEM_EVIDENCE_TERMS
                 if t in (view.summary + " " + " ".join(str(j.get("description","")) for j in view.career)).lower()]
        if kinds:
            out.append(f"career shows {kinds[0]}/{kinds[1] if len(kinds)>1 else 'related'} systems work")
        else:
            out.append("career shows hands-on ML systems work")
    elif v["system_evidence"] >= 0.35:
        out.append("production/applied-ML experience in the career history")

    # Must-have stack.
    if v["must_have_skills"] >= 0.5:
        named = [s.get("name") for s in view.skills
                 if any(rt.strip() in str(s.get("name","")).lower()
                        for rt in (jd_spec.RETRIEVAL_TECH_TERMS + jd_spec.EVAL_FRAMEWORK_TERMS))]
        if named:
            out.append(f"retrieval/eval skills incl. {', '.join(named[:3])}")

    # Experience band — cite "in the JD's band" only when the actual years fall
    # in the stated 5-9 range, so the text never overclaims.
    yoe = view.years_of_experience
    if 5.0 <= yoe <= 9.0:
        out.append(f"{_years(sc)} experience, in the JD's band")
    elif v["experience_fit"] >= 0.55:
        out.append(f"{_years(sc)} experience, near the JD's band")

    # Location.
    if v["location_fit"] >= 0.85:
        out.append(f"based in {view.location}")
    elif sc.view.signals.get("willing_to_relocate"):
        out.append("willing to relocate")

    # Engagement.
    rr = view.signals.get("recruiter_response_rate")
    if isinstance(rr, (int, float)) and rr >= 0.5:
        out.append(f"responsive (recruiter response rate {rr:.0%})")
    return out


def _concerns(sc: ScoredCandidate) -> list[str]:
    """Honest, fact-grounded concern clauses (Stage 4 rewards acknowledged gaps)."""
    v = sc.features.values
    view = sc.view
    out: list[str] = []

    if "keyword_stuffer" in sc.integrity.disqualifier_flags:
        out.append(f"lists AI skills but the title ({view.current_title}) and history show no ML work")
    if "only_consulting" in sc.integrity.disqualifier_flags:
        out.append("entire career at IT-services firms (JD prefers product-company experience)")
    if "title_chaser" in sc.integrity.disqualifier_flags:
        out.append("short tenures suggest title-chasing")
    if "wrong_ml_domain" in sc.integrity.disqualifier_flags:
        out.append("CV/speech-leaning ML without clear NLP/IR exposure")

    if v["role_fit"] < 0.3 and "keyword_stuffer" not in sc.integrity.disqualifier_flags:
        out.append(f"current title ({view.current_title}) is off the target role")
    if v["system_evidence"] < 0.35:
        out.append("limited evidence of ranking/retrieval/recsys systems work")
    if v["experience_fit"] < 0.4:
        out.append(f"{_years(sc)} sits outside the JD's experience band")

    rr = view.signals.get("recruiter_response_rate")
    if isinstance(rr, (int, float)) and rr < jd_spec.LOW_RESPONSE_RATE * 4:  # <0.2
        out.append(f"low recruiter response rate ({rr:.0%}) — availability risk")
    if sc.features.behavioral_components.get("recency", 1.0) < 0.5:
        out.append("not recently active on the platform")
    return out


def generate(sc: ScoredCandidate, rank: int) -> str:
    """Build a 1-2 sentence reasoning whose tone matches the rank."""
    strengths = _strengths(sc)
    concerns = _concerns(sc)

    # Lead with the dominant facts; keep it to ~2 sentences.
    if strengths:
        lead = strengths[0][0].upper() + strengths[0][1:]
        rest = strengths[1:3]
        first = lead + ("; " + "; ".join(rest) if rest else "") + "."
    else:
        first = f"{sc.view.current_title} with {_years(sc)}."

    # Tone tracks rank: strong picks state fit confidently; weak picks foreground
    # the concern. The reasoning is generated AFTER the rank is known, but its
    # content is the same feature values that produced the rank — so it is
    # consistent by construction, not independently invented.
    if concerns:
        if rank <= 30:
            second = f" Concern: {concerns[0]}."
        else:
            # lower ranks: lead the concern, it's the reason they're here not higher
            second = f" Ranked here due to: {concerns[0]}" + (
                f"; {concerns[1]}." if len(concerns) > 1 else ".")
    else:
        second = ""

    return (first + second).strip()
