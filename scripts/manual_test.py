"""
manual_test.py — Manually test the feature pipeline against synthetic candidate profiles.

Run:
    python scripts/manual_test.py

No candidates.jsonl or model download required. Creates 5 archetypal candidates
and prints the full feature breakdown, so you can visually inspect every score
and confirm the new features (premium_pedigree, chronological decay, cold-start)
are working correctly.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redrob_ranker.data import build_view
from redrob_ranker.features import extract
from redrob_ranker.integrity import assess

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic candidate profiles (raw dicts, same schema as candidates.jsonl)
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES = [
    # 1. IDEAL CANDIDATE ── IIT grad, Swiggy, builds ranking systems, open to work
    {
        "candidate_id": "TEST-001",
        "profile": {
            "anonymized_name": "Ideal Candidate",
            "headline": "Senior AI Engineer — Ranking & Retrieval Systems",
            "summary": "Built production recommendation and ranking pipelines at scale using FAISS and BM25.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior Machine Learning Engineer",
            "current_company": "Swiggy",
            "current_company_size": "1001-5000",
            "current_industry": "Internet",
        },
        "career_history": [
            {
                "title": "Senior ML Engineer", "company": "Swiggy", "is_current": True,
                "start_date": "2022-01-01", "duration_months": 29,
                "description": "Built recommendation and ranking systems using Elasticsearch, FAISS, BM25. A/B testing for retrieval pipelines.",
            },
            {
                "title": "ML Engineer", "company": "Flipkart", "is_current": False,
                "start_date": "2019-06-01", "duration_months": 30,
                "description": "Developed recsys and search ranking. Implemented embedding-based retrieval with Qdrant.",
            },
        ],
        "education": [{"school": "IIT Bombay", "degree": "B.Tech", "field": "Computer Science"}],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 15, "duration_months": 72},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 30},
            {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 3, "duration_months": 24},
            {"name": "BM25", "proficiency": "advanced", "endorsements": 2, "duration_months": 18},
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.95,
            "last_active_date": "2026-06-20",
            "notice_period_days": 15,
            "willing_to_relocate": True,
            "interview_completion_rate": 0.9,
            "verified_email": True, "verified_phone": True, "linkedin_connected": True,
            "github_activity_score": 75,
            "skill_assessment_scores": {"machine learning": 92, "nlp": 88},
        },
    },

    # 2. KEYWORD STUFFER ── HR Manager who lists AI skills but wrong title
    {
        "candidate_id": "TEST-002",
        "profile": {
            "anonymized_name": "Keyword Stuffer",
            "headline": "HR Manager | AI Enthusiast | Machine Learning | NLP | LLM",
            "summary": "Passionate about AI, Machine Learning, Deep Learning, NLP, LLM, FAISS, Elasticsearch.",
            "location": "Bangalore",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "HR Manager",
            "current_company": "TCS",
            "current_company_size": "10000+",
            "current_industry": "Human Resources",
        },
        "career_history": [
            {
                "title": "HR Manager", "company": "TCS", "is_current": True,
                "start_date": "2021-01-01", "duration_months": 65,
                "description": "Managed recruitment drives and employee engagement. Interested in AI tools for hiring.",
            },
        ],
        "education": [{"school": "Mumbai University", "degree": "MBA", "field": "HR"}],
        "skills": [
            {"name": "Machine Learning", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
            {"name": "NLP", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
            {"name": "FAISS", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
            {"name": "Elasticsearch", "proficiency": "beginner", "endorsements": 0, "duration_months": 0},
        ],
        "redrob_signals": {
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.3,
            "last_active_date": "2025-11-01",
            "notice_period_days": 90,
            "willing_to_relocate": False,
            "interview_completion_rate": 0.4,
            "verified_email": True, "verified_phone": False, "linkedin_connected": False,
            "github_activity_score": -1,
            "skill_assessment_scores": {"machine learning": 25, "nlp": 18},
        },
    },

    # 3. COLD-START ── Good title (AI Engineer), but sparse profile, should get inferred skills
    {
        "candidate_id": "TEST-003",
        "profile": {
            "anonymized_name": "Cold Start AI Engineer",
            "headline": "AI Engineer",
            "summary": "",
            "location": "Noida",
            "country": "India",
            "years_of_experience": 4.0,
            "current_title": "AI Engineer",
            "current_company": "A Startup",
            "current_company_size": "11-50",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "title": "AI Engineer", "company": "A Startup", "is_current": True,
                "start_date": "2023-01-01", "duration_months": 18,
                "description": "Built recommendation and ranking pipelines.",
            },
        ],
        "education": [{"school": "NIT Trichy", "degree": "B.Tech", "field": "ECE"}],
        "skills": [],  # No skills listed — triggers cold-start enrichment
        "redrob_signals": {
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.7,
            "last_active_date": "2026-06-10",
            "notice_period_days": 30,
            "willing_to_relocate": True,
            "interview_completion_rate": 0.8,
            "verified_email": True, "verified_phone": True, "linkedin_connected": False,
            "github_activity_score": 50,
            "skill_assessment_scores": {},
        },
    },

    # 4. HOLLOW PROFILE (should be BLOCKED by Cold-Start Gatekeeper)
    {
        "candidate_id": "TEST-004",
        "profile": {
            "anonymized_name": "Hollow Keyword Stuffer",
            "headline": "AI Engineer | NLP | LLM",
            "summary": "",
            "location": "Delhi",
            "country": "India",
            "years_of_experience": 0,
            "current_title": "AI Engineer",
            "current_company": "",
            "current_company_size": "",
            "current_industry": "",
        },
        "career_history": [],  # Totally empty career — gatekeeper MUST block inference
        "education": [],
        "skills": [],
        "redrob_signals": {},
    },

    # 5. OUTDATED EXPERT ── Built recsys 8 years ago, now in a different role
    {
        "candidate_id": "TEST-005",
        "profile": {
            "anonymized_name": "Outdated Expert",
            "headline": "Product Manager",
            "summary": "Transitioned from ML to Product Management.",
            "location": "Hyderabad",
            "country": "India",
            "years_of_experience": 11.0,
            "current_title": "Product Manager",
            "current_company": "Some Corp",
            "current_company_size": "501-1000",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "title": "Product Manager", "company": "Some Corp", "is_current": True,
                "start_date": "2021-01-01", "duration_months": 66,
                "description": "Managing product roadmap, user research, stakeholder alignment.",
            },
            {
                "title": "ML Engineer", "company": "OldCo", "is_current": False,
                "start_date": "2015-01-01", "duration_months": 48,
                "description": "Built recommendation and ranking systems using collaborative filtering and BM25.",
            },
        ],
        "education": [{"school": "BITS Pilani", "degree": "B.E.", "field": "CS"}],
        "skills": [
            {"name": "Python", "proficiency": "intermediate", "endorsements": 3, "duration_months": 48},
        ],
        "redrob_signals": {
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.5,
            "last_active_date": "2025-10-01",
            "notice_period_days": 90,
            "willing_to_relocate": False,
            "interview_completion_rate": 0.6,
            "verified_email": True, "verified_phone": False, "linkedin_connected": True,
            "github_activity_score": -1,
            "skill_assessment_scores": {},
        },
    },
]

LABELS = {
    "TEST-001": "[IDEAL] IIT+Swiggy, active ranking builder",
    "TEST-002": "[STUFFER] HR Manager, wrong role",
    "TEST-003": "[COLD-START] Sparse profile, good title+NIT",
    "TEST-004": "[HOLLOW] 0 exp + 0 history => gatekeeper should BLOCK cold-start",
    "TEST-005": "[OUTDATED] BITS grad, built recsys 8 years ago, now PM",
}


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  MANUAL FEATURE TEST - Redrob Ranker")
    print("=" * 70)

    for raw in CANDIDATES:
        cid = raw["candidate_id"]
        view = build_view(raw)
        fb = extract(view)
        rep = assess(view)

        # Cold-start check: how many inferred skills got injected?
        inferred_count = sum(1 for s in view.skills if s.get("is_inferred"))

        print(f"\n{'-' * 70}")
        print(f"  {cid} | {LABELS[cid]}")
        print(f"  Title: {view.current_title}  |  Exp: {view.years_of_experience}y  |  Location: {view.location}")
        print(f"  Inferred skills injected: {inferred_count}")
        print(f"  Honeypot: {rep.is_honeypot}  |  Disqualifiers: {list(rep.disqualifier_flags.keys()) or 'None'}")
        print()
        print(f"  {'FEATURE':<22} {'SCORE':>7}")
        print(f"  {'-' * 30}")
        for k, val in fb.values.items():
            bar = "#" * int(val * 20)
            print(f"  {k:<22} {val:>6.3f}  {bar}")
        print(f"\n  {'behavioral_mult':<22} {fb.behavioral_mult:>6.3f}")

    print(f"\n{'=' * 70}")
    print("  KEY THINGS TO VERIFY:")
    print("  1. TEST-001 (Ideal):        All features HIGH, pedigree = 1.0 (IIT+Swiggy)")
    print("  2. TEST-002 (Stuffer):      role_fit LOW, must_have_skills LOW (0 endorsements)")
    print("  3. TEST-003 (Cold-Start):   Inferred skills > 0, pedigree = 0.5 (NIT)")
    print("  4. TEST-004 (Hollow):       Inferred skills = 0 (gatekeeper blocked it)")
    print("  5. TEST-005 (Outdated):     system_evidence LOW (decay penalized old roles)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
