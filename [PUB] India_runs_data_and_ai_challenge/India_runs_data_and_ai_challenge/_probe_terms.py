"""How often do system-evidence terms fire, and which terms actually appear in
the career descriptions of genuinely-relevant (ML/DS/data-titled) candidates?"""
import json, collections, re

SYS = ("recommendation","recommender","ranking","search relevance","retrieval",
       "semantic search","vector search","embedding","personalization",
       "information retrieval","learning to rank","matching system","feed ranking")
# candidate broader terms to consider adding
EXTRA = ("data pipeline","feature pipeline","ml model","machine learning","model training",
         "deployed","production","recommendation system","relevance","nlp","neural network",
         "deep learning","predictive model","classification model","scoring model","a/b test")

relevant_titles = ("ml engineer","machine learning","ai engineer","ai research","data scientist",
                   "data engineer","analytics engineer","backend engineer","software engineer",
                   "nlp","research engineer","full stack")
fire_sys = 0; rel = 0
term_freq = collections.Counter()
extra_freq = collections.Counter()
with open("candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line); tl = c["profile"]["current_title"].lower()
        if not any(t in tl for t in relevant_titles): continue
        rel += 1
        text = " ".join(j.get("description","") for j in c.get("career_history",[])).lower()
        text += " " + c["profile"].get("summary","").lower()
        if any(t in text for t in SYS): fire_sys += 1
        for t in SYS:
            if t in text: term_freq[t]+=1
        for t in EXTRA:
            if t in text: extra_freq[t]+=1
print("relevant-titled candidates:", rel)
print("of those, system_evidence(SYS) fires for:", fire_sys, f"({100*fire_sys/rel:.1f}%)")
print("\nSYS term frequencies (within relevant):")
for t,ct in term_freq.most_common(): print(f"  {ct:5d}  {t}")
print("\nEXTRA candidate-term frequencies (within relevant):")
for t,ct in extra_freq.most_common(): print(f"  {ct:5d}  {t}")
