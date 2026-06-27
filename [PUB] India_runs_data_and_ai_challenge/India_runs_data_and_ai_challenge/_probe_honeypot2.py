import json, datetime as dt
REF = dt.date(2026, 6, 1)
def pdate(s):
    try: return dt.date.fromisoformat(s[:10])
    except Exception: return None

# Refined, high-precision impossibility checks. We want signatures that fire on
# ~80 profiles, not 13k. Examine each candidate against several and print the union
# plus a couple of full examples.
strong = set(); reasons = {}
samples = []
n = 0
with open("candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line); n += 1
        cid = c["candidate_id"]; p = c["profile"]; yoe = p.get("years_of_experience", 0) or 0
        rs = []
        # 1. high proficiency, zero duration
        if any(s.get("proficiency") in ("advanced","expert") and s.get("duration_months",1)==0 for s in c.get("skills",[])):
            rs.append("expert_skill_0_months")
        # 2. a single job longer than the entire claimed career (+6mo slack)
        if any(j.get("duration_months",0) > yoe*12 + 6 for j in c.get("career_history",[])):
            rs.append("job_longer_than_career")
        # 3. start_date AFTER end_date, or end before start
        for j in c.get("career_history",[]):
            sd, ed = pdate(j.get("start_date")), pdate(j.get("end_date"))
            if sd and ed and ed < sd:
                rs.append("end_before_start"); break
        # 4. duration_months inconsistent with start/end by a lot (>9 months off)
        for j in c.get("career_history",[]):
            sd, ed = pdate(j.get("start_date")), pdate(j.get("end_date"))
            dm = j.get("duration_months")
            if sd and ed and dm is not None:
                actual = (ed.year-sd.year)*12 + (ed.month-sd.month)
                if abs(actual - dm) > 9:
                    rs.append("duration_mismatch"); break
        # 5. skill duration exceeds career by a LOT (>3 years beyond yoe) - tighter than before
        if any(s.get("duration_months",0) > yoe*12 + 36 for s in c.get("skills",[])):
            rs.append("skill_way_longer_than_career")
        if rs:
            strong.add(cid)
            for r in rs: reasons[r] = reasons.get(r,0)+1
            if len(samples) < 3 and "duration_mismatch" not in rs:
                samples.append((cid, rs, yoe, c["profile"]["current_title"]))
print("total", n, "union strong-impossible:", len(strong))
for r,ct in sorted(reasons.items(), key=lambda x:-x[1]): print(f"  {r:30s} {ct}")
print()
for cid, rs, yoe, t in samples:
    print(cid, t, "yoe=",yoe, rs)
