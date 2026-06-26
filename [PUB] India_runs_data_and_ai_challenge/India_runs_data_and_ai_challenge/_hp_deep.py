"""Deep honeypot signature hunt. Goal: find signatures that, unioned, approach ~80
WITHOUT firing on thousands of normal profiles. Test many impossibility types."""
import json, datetime as dt, collections
REF = dt.date(2026, 6, 1)
def pdate(s):
    try: return dt.date.fromisoformat(s[:10])
    except Exception: return None

sigs = collections.defaultdict(set)   # signature -> set of cand ids
all_cids = []
with open("candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line); cid = c["candidate_id"]; all_cids.append(cid)
        p = c["profile"]; yoe = p.get("years_of_experience", 0) or 0
        skills = c.get("skills", []); career = c.get("career_history", []); edu = c.get("education", [])

        # A. expert/advanced skill, 0 months
        if any(s.get("proficiency") in ("advanced","expert") and s.get("duration_months",1)==0 for s in skills):
            sigs["A_expert_0mo"].add(cid)
        # B. single job > whole career
        if any(j.get("duration_months",0) > yoe*12 + 6 for j in career):
            sigs["B_job>career"].add(cid)
        # C. job end before start
        for j in career:
            sd, ed = pdate(j.get("start_date")), pdate(j.get("end_date"))
            if sd and ed and ed < sd: sigs["C_end<start"].add(cid); break
        # D. duration_months vs (end-start) mismatch > 12 months
        for j in career:
            sd, ed = pdate(j.get("start_date")), pdate(j.get("end_date"))
            dm = j.get("duration_months")
            if sd and ed and dm is not None:
                actual = (ed.year-sd.year)*12 + (ed.month-sd.month)
                if abs(actual - dm) > 12: sigs["D_dur_mismatch"].add(cid); break
        # E. earliest career start implies age/experience way beyond yoe
        starts = [pdate(j.get("start_date")) for j in career if pdate(j.get("start_date"))]
        if starts:
            span = (REF - min(starts)).days/365.25
            if span > yoe + 4: sigs["E_career_span>yoe"].add(cid)
        # F. education end_year after a job that started before graduation by years
        #    (i.e., working senior role long before finishing degree) -- impossible-ish
        grad_years = [e.get("end_year") for e in edu if isinstance(e.get("end_year"), int)]
        if grad_years and starts:
            earliest_job = min(starts).year
            latest_grad = max(grad_years)
            # started working 5+ yrs before finishing ANY listed degree AND only one degree
            if earliest_job <= latest_grad - 6: sigs["F_job_before_grad"].add(cid)
        # G. education end_year < start_year
        for e in edu:
            sy, ey = e.get("start_year"), e.get("end_year")
            if isinstance(sy,int) and isinstance(ey,int) and ey < sy: sigs["G_edu_end<start"].add(cid); break
        # H. is_current True but has end_date
        for j in career:
            if j.get("is_current") and j.get("end_date"): sigs["H_current_with_end"].add(cid); break
        # I. skill duration_months > career span by a lot
        if starts:
            career_span_mo = (REF - min(starts)).days/30.4
            if any(s.get("duration_months",0) > career_span_mo + 24 for s in skills):
                sigs["I_skill>careerspan"].add(cid)
        # J. sum of NON-overlapping job durations > career span (serial jobs can't overlap)
        # approximate: sum durations vs span
        if starts and len(career) >= 2:
            ends = [pdate(j.get("end_date")) or REF for j in career]
            span_mo = (max(ends) - min(starts)).days/30.4
            total = sum(j.get("duration_months",0) for j in career)
            if total > span_mo + 18: sigs["J_overlap_impossible"].add(cid)

print("total", len(all_cids))
for k in sorted(sigs): print(f"  {k:22s} {len(sigs[k])}")
union = set().union(*sigs.values())
print("UNION all:", len(union))
# union of just the "tight" ones
tight = sigs["A_expert_0mo"]|sigs["B_job>career"]|sigs["C_end<start"]|sigs["D_dur_mismatch"]|sigs["G_edu_end<start"]|sigs["H_current_with_end"]
print("UNION tight (A,B,C,D,G,H):", len(tight))
