"""
Unit tests for the ranker's core invariants. Run: pytest -q

These lock in the behaviours we rely on:
  * honeypot detection fires on the spec's exact impossible-profile examples,
  * features stay in [0,1] and reward on-role over keyword-stuffers,
  * the submission writer enforces every validator rule.
They double as executable documentation of the design decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker.candidate_generation import passes_gate
from redrob_ranker.data import build_view
from redrob_ranker.features import FEATURE_NAMES, extract
from redrob_ranker.integrity import assess
from redrob_ranker.ranker import RankRow
from redrob_ranker.submission import validate_rows


def _base_candidate(**over) -> dict:
    """A minimal valid candidate; override fields per test."""
    c = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer",
            "summary": "Built recommendation and ranking systems in production.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_company": "Acme Product Co",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "Acme Product Co", "title": "ML Engineer",
            "start_date": "2021-01-01", "end_date": None, "duration_months": 40,
            "is_current": True, "industry": "Software", "company_size": "201-500",
            "description": "Built a recommendation system with embeddings and FAISS; "
                           "evaluated ranking with NDCG and ran A/B tests in production.",
        }],
        "education": [],
        "skills": [
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 30},
            {"name": "NDCG", "proficiency": "advanced", "endorsements": 5, "duration_months": 24},
        ],
        "certifications": [], "languages": [],
        "redrob_signals": {
            "last_active_date": "2026-05-20", "recruiter_response_rate": 0.8,
            "open_to_work_flag": True, "verified_email": True, "verified_phone": True,
            "linkedin_connected": True, "interview_completion_rate": 0.9,
            "willing_to_relocate": True,
        },
    }
    prof = over.pop("profile", {})
    c["profile"].update(prof)
    c.update(over)
    return c


# --------------------------------------------------------------------------- #
# Honeypot detection
# --------------------------------------------------------------------------- #

def test_honeypot_zero_duration_expert_skill():
    c = _base_candidate()
    c["skills"].append({"name": "Kubernetes", "proficiency": "expert", "endorsements": 0, "duration_months": 0})
    rep = assess(build_view(c))
    assert rep.is_honeypot
    assert any("0 months" in r for r in rep.honeypot_reasons)


def test_honeypot_job_longer_than_career():
    c = _base_candidate(profile={"years_of_experience": 8.0})
    c["career_history"][0]["duration_months"] = 8 * 12 + 60  # impossible vs 8y
    rep = assess(build_view(c))
    assert rep.is_honeypot


def test_clean_profile_is_not_honeypot():
    rep = assess(build_view(_base_candidate()))
    assert not rep.is_honeypot


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def test_features_in_unit_range():
    fb = extract(build_view(_base_candidate()))
    for name in FEATURE_NAMES:
        assert 0.0 <= fb.values[name] <= 1.0, name
    assert 0.55 <= fb.behavioral_mult <= 1.0


def test_on_role_beats_keyword_stuffer():
    on_role = extract(build_view(_base_candidate()))
    stuffer = _base_candidate(profile={
        "current_title": "Graphic Designer", "headline": "Graphic Designer",
        "summary": "Designer who lists many AI tools.",
    })
    stuffer["career_history"][0]["title"] = "Graphic Designer"
    stuffer["career_history"][0]["description"] = "Designed brand assets and marketing collateral."
    fb_stuffer = extract(build_view(stuffer))
    assert on_role.values["role_fit"] > fb_stuffer.values["role_fit"]


def test_keyword_stuffer_flagged():
    c = _base_candidate(profile={
        "current_title": "Accountant", "headline": "Accountant",
        "summary": "Finance professional.",
    })
    c["career_history"][0]["title"] = "Accountant"
    c["career_history"][0]["description"] = "Managed ledgers, audits and tax filing."
    c["skills"] = [
        {"name": "RAG", "proficiency": "expert", "endorsements": 1, "duration_months": 5},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 1, "duration_months": 5},
        {"name": "Embeddings", "proficiency": "expert", "endorsements": 1, "duration_months": 5},
        {"name": "LoRA", "proficiency": "expert", "endorsements": 1, "duration_months": 5},
    ]
    rep = assess(build_view(c))
    assert "keyword_stuffer" in rep.disqualifier_flags


# --------------------------------------------------------------------------- #
# Stage 1 — candidate generation gate
# --------------------------------------------------------------------------- #

def _gate(c: dict) -> bool:
    v = build_view(c)
    return passes_gate(v, extract(v), assess(v)).eligible


def test_gate_admits_on_role_systems_builder():
    assert _gate(_base_candidate()) is True


def test_gate_rejects_off_role_no_evidence():
    c = _base_candidate(profile={
        "current_title": "Accountant", "headline": "Accountant", "summary": "Finance pro.",
    })
    c["career_history"][0]["title"] = "Accountant"
    c["career_history"][0]["description"] = "Managed ledgers and tax filing."
    c["skills"] = [{"name": "Excel", "proficiency": "advanced", "endorsements": 5, "duration_months": 40}]
    assert _gate(c) is False


def test_gate_rejects_honeypot():
    c = _base_candidate()
    c["skills"].append({"name": "Spark", "proficiency": "expert", "endorsements": 0, "duration_months": 0})
    assert _gate(c) is False


def test_gate_admits_adjacent_with_strong_systems_evidence():
    # Adjacent title (Data Engineer) but career clearly shows recsys/ranking work.
    c = _base_candidate(profile={
        "current_title": "Data Engineer", "headline": "Data Engineer",
        "summary": "Built recommendation and ranking systems and retrieval pipelines.",
    })
    c["career_history"][0]["title"] = "Data Engineer"
    c["career_history"][0]["description"] = (
        "Built a recommendation system and search ranking pipeline with embeddings and retrieval."
    )
    assert _gate(c) is True


# --------------------------------------------------------------------------- #
# Submission validation
# --------------------------------------------------------------------------- #

def _good_rows() -> list[RankRow]:
    return [
        RankRow(f"CAND_{i:07d}", i, 1.0 - i * 0.001, f"reason {i}")
        for i in range(1, 101)
    ]


def test_valid_rows_pass():
    assert validate_rows(_good_rows()) == []


def test_increasing_score_rejected():
    rows = _good_rows()
    rows[5].score = 5.0  # rank 6 scores higher than rank 1
    errs = validate_rows(rows)
    assert any("increase" in e for e in errs)


def test_wrong_count_rejected():
    assert validate_rows(_good_rows()[:99]) != []


def test_duplicate_id_rejected():
    rows = _good_rows()
    rows[1].candidate_id = rows[0].candidate_id
    assert any("duplicate candidate_id" in e for e in validate_rows(rows))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
