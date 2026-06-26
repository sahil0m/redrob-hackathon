import json, collections, datetime

aititles = collections.Counter()
honeypot = []
stuffer = []
RELEVANT_KW = ['ml engineer','machine learning','ai engineer','ai research','data scientist','nlp','research engineer']
AI_SKILLS = {'rag','embeddings','vector search','pinecone','weaviate','qdrant','milvus','faiss',
             'elasticsearch','opensearch','sentence-transformers','bge','e5','llm','fine-tuning llms',
             'lora','qlora','peft','xgboost','learning to rank','ndcg','retrieval','semantic search',
             'transformers','pytorch','tensorflow','bm25','recommendation'}

def parse(d):
    try: return datetime.date.fromisoformat(d)
    except: return None

n = 0
with open('candidates.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line); p = c['profile']; t = p['current_title']; n += 1
        tl = t.lower()
        if any(k in tl for k in RELEVANT_KW):
            aititles[t] += 1
        skl = c['skills']
        yoe = p['years_of_experience']
        # honeypot signal A: high proficiency, 0 months used
        imposs_skill = any(s.get('proficiency') in ('expert','advanced') and s.get('duration_months', 1) == 0 for s in skl)
        # honeypot signal B: a job tenure that exceeds time since the company could exist
        # approximate via: a single role duration > yoe*12 (impossible) OR start_date before (now - yoe - slack)
        tenure_impossible = False
        for j in c['career_history']:
            if j.get('duration_months', 0) > yoe * 12 + 6:
                tenure_impossible = True
        if imposs_skill or tenure_impossible:
            honeypot.append((c['candidate_id'], t, yoe, imposs_skill, tenure_impossible))
        # keyword stuffer: many AI skills as 'expert' but title totally unrelated & no AI in career descriptions
        ai_skill_ct = sum(1 for s in skl if s['name'].lower() in AI_SKILLS)
        career_text = ' '.join(j.get('description','') for j in c['career_history']).lower()
        career_has_ai = any(k in career_text for k in ['machine learning','ml model','embedding','retrieval','ranking','recommendation','nlp','neural','llm'])
        irrelevant_title = not any(k in tl for k in ['engineer','developer','scientist','ml','ai','data','research','analyst'])
        if ai_skill_ct >= 5 and irrelevant_title and not career_has_ai:
            stuffer.append((c['candidate_id'], t, ai_skill_ct, yoe))

print('TOTAL', n)
print('\n=== AI/ML/DS TITLES ===')
for t, ct in aititles.most_common(40):
    print(f'{ct:5d}  {t}')
print('\n=== honeypot-ish flagged:', len(honeypot))
for e in honeypot[:10]: print('  ', e)
print('\n=== keyword-stuffer-ish flagged:', len(stuffer))
for e in stuffer[:10]: print('  ', e)
