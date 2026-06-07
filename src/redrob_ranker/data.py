"""
data.py — Loading candidates and turning each raw record into a flat, typed view.

Two responsibilities:
  1. Stream candidates.jsonl (or .gz) without loading 465 MB into memory at once.
  2. Derive a `CandidateView` per record: a small, flat dataclass holding exactly
     the fields the scorer and honeypot detector need, plus two free-text blobs
     (one for dense embedding, one for BM25). Keeping derivation in one place
     means the scorer never re-parses raw JSON and the reasoning layer reads the
     same numbers the score was computed from.

We are defensive about the data because the dataset deliberately contains
malformed / contradictory profiles (that is the point of the honeypots). Missing
or out-of-range fields degrade gracefully rather than crash.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Cap on words fed to the embedder. ~256 words ≈ within e5's 512-token window
# while front-loading title/summary/recent roles (the JD-decisive content).
DENSE_WORD_CAP = 256


def _open_maybe_gzip(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_raw(path: str | Path) -> Iterator[dict]:
    """Yield one parsed candidate dict per non-empty JSONL line."""
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A truly unparseable line is itself a kind of corruption; skip
                # it rather than abort the whole 100k run.
                continue


def _parse_date(s) -> dt.date | None:
    if not isinstance(s, str):
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


@dataclass
class CandidateView:
    """A flattened, defensively-typed projection of one candidate record."""

    candidate_id: str

    # --- profile ---
    name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

    # --- career history (kept structured; the scorer reasons over tenure/order) ---
    career: list[dict] = field(default_factory=list)

    # --- education ---
    education: list[dict] = field(default_factory=list)

    # --- skills: name -> (proficiency, endorsements, duration_months) ---
    skills: list[dict] = field(default_factory=list)

    certifications: list[dict] = field(default_factory=list)
    languages: list[dict] = field(default_factory=list)

    # --- redrob behavioral signals (flattened, the 23 fields) ---
    signals: dict = field(default_factory=dict)

    # --- derived free text ---
    # dense_text: rich semantic blob for the embedding model.
    # sparse_text: lowercased blob for BM25 (favors exact term overlap).
    dense_text: str = ""
    sparse_text: str = ""

    @property
    def proficiency_order(self) -> dict[str, int]:
        return {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


def _safe(d: dict, key: str, default):
    v = d.get(key, default)
    return v if v is not None else default


def build_view(raw: dict) -> CandidateView:
    """Convert one raw candidate dict into a CandidateView. Never raises."""
    p = raw.get("profile", {}) or {}
    sig = raw.get("redrob_signals", {}) or {}
    career = raw.get("career_history", []) or []
    edu = raw.get("education", []) or []
    skills = raw.get("skills", []) or []

    summary = str(_safe(p, "summary", ""))
    headline = str(_safe(p, "headline", ""))
    title = str(_safe(p, "current_title", ""))

    # Dense text: human-readable, in the order a recruiter would read it. The
    # career descriptions carry the strongest "what did they actually build"
    # signal. We front-load the most JD-relevant content (title, summary, the two
    # most recent roles) and cap the total to ~DENSE_WORD_CAP words: e5 truncates
    # at 512 tokens anyway, so an unbounded blob just gets cut arbitrarily by the
    # tokenizer. Front-loading keeps the decisive signal inside the window AND
    # roughly halves embedding time on CPU (measured).
    career_titles = " ".join(str(_safe(j, "title", "")) for j in career)
    skill_names = ", ".join(str(_safe(s, "name", "")) for s in skills)
    # Most recent roles first (career_history is chronological; current role has
    # is_current=True). Sort current-first, then by start_date desc if present.
    ordered = sorted(
        career,
        key=lambda j: (bool(j.get("is_current")), str(j.get("start_date", ""))),
        reverse=True,
    )
    recent_descs = " ".join(str(_safe(j, "description", "")) for j in ordered[:3])
    dense_parts = [
        f"Title: {title}.",
        f"Headline: {headline}.",
        f"Summary: {summary}",
        f"Experience: {_safe(p, 'years_of_experience', 0)} years.",
        f"Past roles: {career_titles}.",
        f"What they built: {recent_descs}",
        f"Skills: {skill_names}.",
    ]
    dense_text = " ".join(dense_parts)
    # Hard word cap to bound tokenizer work and keep the JD-relevant head.
    words = dense_text.split()
    if len(words) > DENSE_WORD_CAP:
        dense_text = " ".join(words[:DENSE_WORD_CAP])
    # full career text retained for the sparse/lexical side (BM25 wants it all)
    career_descs = " ".join(str(_safe(j, "description", "")) for j in career)

    # Sparse text: same content, lowercased, no field labels — BM25 wants raw
    # term frequencies.
    sparse_text = " ".join(
        [title, headline, summary, career_titles, career_descs, skill_names]
    ).lower()

    return CandidateView(
        candidate_id=str(raw.get("candidate_id", "")),
        name=str(_safe(p, "anonymized_name", "")),
        headline=headline,
        summary=summary,
        location=str(_safe(p, "location", "")),
        country=str(_safe(p, "country", "")),
        years_of_experience=float(_safe(p, "years_of_experience", 0.0) or 0.0),
        current_title=title,
        current_company=str(_safe(p, "current_company", "")),
        current_company_size=str(_safe(p, "current_company_size", "")),
        current_industry=str(_safe(p, "current_industry", "")),
        career=career,
        education=edu,
        skills=skills,
        certifications=raw.get("certifications", []) or [],
        languages=raw.get("languages", []) or [],
        signals=sig,
        dense_text=dense_text,
        sparse_text=sparse_text,
    )


def load_views(path: str | Path) -> list[CandidateView]:
    """Load all candidates as CandidateView objects."""
    return [build_view(raw) for raw in stream_raw(path)]


# Re-export the date parser so the scorer/honeypot detector share one implementation.
parse_date = _parse_date
