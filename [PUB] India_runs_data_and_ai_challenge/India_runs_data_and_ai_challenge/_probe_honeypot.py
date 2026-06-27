"""Probe honeypot signatures precisely against the full pool to calibrate rules.
The spec says ~80 honeypots with 'subtly impossible' profiles. We test several
independent impossibility checks and see how many each catches and how they overlap."""
import json, datetime as dt

REF = dt.date(2026, 6, 1)

def pdate(s):
    try: return dt.date.fromisoformat(s[:10])
    except Exception: return None

checks = {
    "skill_dur0_highprof": 0,   # advanced/expert proficiency but duration_months == 0
    "skill_dur_gt_yoe": 0,      # a skill used longer than the person has worked
    "job_dur_gt_yoe": 0,        # a single job longer than total career
    "sum_job_dur_gt_yoe": 0,    # sum of tenures >> yoe (overlap impossible if serial)
    "edu_after_career_start": 0,
    "tenure_at_young_company": 0,  # job duration > company age is approximated by start before founding — can't know founding; skip
    "endorse_gt_received": 0,
    "negative_or_huge": 0,
}
flagged_ids = {k: [] for k in checks}
any_flag = set()
n = 0
with open("candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line); n += 1
        cid = c["candidate_id"]; p = c["profile"]; yoe = p.get("years_of_experience", 0) or 0
        skills = c.get("skills", [])
        career = c.get("career_history", [])

        for s in skills:
            if s.get("proficiency") in ("advanced", "expert") and s.get("duration_months", 1) == 0:
                checks["skill_dur0_highprof"] += 1; flagged_ids["skill_dur0_highprof"].append(cid); any_flag.add(cid); break
        for s in skills:
            if s.get("duration_months", 0) > yoe * 12 + 6:
                checks["skill_dur_gt_yoe"] += 1; flagged_ids["skill_dur_gt_yoe"].append(cid); any_flag.add(cid); break
        for j in career:
            if j.get("duration_months", 0) > yoe * 12 + 6:
                checks["job_dur_gt_yoe"] += 1; flagged_ids["job_dur_gt_yoe"].append(cid); any_flag.add(cid); break
        total = sum(j.get("duration_months", 0) for j in career)
        if total > yoe * 12 + 24:
            checks["sum_job_dur_gt_yoe"] += 1; flagged_ids["sum_job_dur_gt_yoe"].append(cid); any_flag.add(cid)
        # job started before the person was plausibly working (yoe vs earliest start)
        starts = [pdate(j.get("start_date")) for j in career if pdate(j.get("start_date"))]
        if starts:
            earliest = min(starts)
            years_since_earliest = (REF - earliest).days / 365.25
            if years_since_earliest > yoe + 3:  # career spans more years than claimed experience+slack
                checks["edu_after_career_start"] += 1; flagged_ids["edu_after_career_start"].append(cid); any_flag.add(cid)

print("total", n)
for k, v in checks.items():
    print(f"  {k:24s} {v}")
print("UNION any flag:", len(any_flag))
print("\nsamples skill_dur0_highprof:", flagged_ids["skill_dur0_highprof"][:8])
