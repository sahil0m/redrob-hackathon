# Redrob Candidate Ranker

Intent-aware top-100 candidate ranking for the Redrob *Intelligent Candidate
Discovery & Ranking* challenge. The system reads the **"Senior AI Engineer —
Founding Team"** job description, models what the role *means* (not which keywords
it contains), and ranks the 100,000-candidate pool best-fit-first.

It is **CPU-only, offline, and reproducible**: precompute embeddings once, then
the ranking step produces `submission.csv` within the spec's 5-minute / 16 GB /
no-network budget.

---

## TL;DR — reproduce the submission

```bash
# 0. install (CPU torch + the rest)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 1. one-time, offline pre-computation of candidate embeddings (~25-35 min CPU)
python scripts/precompute.py --candidates ./candidates.jsonl

# 2. the ranking step — this is the single reproduce command (Stage-3 reproduced)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Step 2 runs in well under 5 minutes on CPU because the expensive embedding work
is already cached. If you skip step 1, `rank.py` still runs end-to-end using a
BM25-only semantic signal (lower quality, zero setup).

---

## Measured results (full 100k pool, CPU)

We don't have the hidden ground truth, so we built an **independent evaluation
harness** ([`eval_labels.py`](src/redrob_ranker/eval_labels.py) +
[`metrics.py`](src/redrob_ranker/metrics.py)) that grades every candidate with
discrete JD rules — derived through a *different reasoning path* than the
ranker's features, so the evaluation is not circular — and computes the exact
competition metrics.

| Metric | Full 100k |
|---|---|
| NDCG@10 | **1.000** |
| NDCG@50 | **1.000** |
| P@10 | **1.000** |
| MAP | 0.118¹ |
| **composite** | **0.868** |
| honeypots in top-100 | **0** (DQ threshold is >10) |
| rank-step runtime | **~46 s** (budget: 5 min) |

¹ MAP is mathematically capped on a top-100 task: there are ~847 relevant
candidates (tier 3+) but only 100 slots, so even a perfect ranker cannot retrieve
the other ~747. MAP is 15% of the composite; the head metrics (NDCG@10/50 = 80%)
are saturated with genuine bullseye candidates.

**Validated, not asserted:**
- *LTR earns its place* — measurably beats the linear blend (composite 0.931 vs
  0.924, NDCG@50 0.985 vs 0.965 on a 20k slice).
- *Not overfit to our labels* — re-graded under a stricter, independent
  "recruiter" policy we did not design the ranker around, it still scores 0.940.
- *Dense retrieval does real work* — it agrees with BM25 on *which* candidates are
  top-tier (18/20 overlap) but **re-orders them** (only 6/20 same position),
  breaking ties among bullseye candidates by semantic intent — exactly where
  NDCG@10 points are won against finer ground truth.

The actual top-5 the system returns:

```
#1 Senior Machine Learning Engineer — ranking/retrieval systems; Elasticsearch, Qdrant
#2 Search Engineer                  — recommendation/ranking; Milvus, Weaviate, BM25
#3 Senior NLP Engineer              — recsys; OpenSearch, FAISS, Embeddings
#4 Lead AI Engineer                 — ranking/retrieval; Elasticsearch, Embeddings, BM25
#5 Recommendation Systems Engineer  — recsys; Qdrant, Embeddings, Milvus
```

Reproduce the evaluation: `python scripts/evaluate.py --candidates ./candidates.jsonl`

## Why this design

The challenge is explicitly built to punish keyword matching. The provided
`sample_submission.csv` is itself the trap — it ranks HR Managers and Graphic
Designers at the top because they *list* AI skills. The JD tells participants
directly: a Marketing Manager with every AI keyword is **not** a fit; a data
engineer who *built a recommender at a product company* **is**, even without the
buzzwords. So every design choice here optimises for **reading the profile**, not
the vocabulary.

### Tailored for the Indian Tech Ecosystem ("Built for India" Localization)

Redrob AI is building a localized operating system for hiring in India. Our system architecture directly aligns with this mission by tailoring features and thresholds to the unique patterns of the Indian talent pool:

1. **IT-Services-to-Product Transition Nuances:** A vast majority of candidates in the 100k pool have entire careers at large Indian IT services/consulting firms (TCS, Infosys, Wipro, Cognizant, Accenture, HCL, etc.). While the JD explicitly prefers product-company backgrounds, we do not apply a blunt filter. Instead, we use a calibrated penalty (`only_consulting` x0.55) that *only* triggers if the candidate's *entire* career is consulting, while granting full credit if they have successfully transitioned or built production-scale systems in their career.
2. **Indian Tech Hub Geography:** The JD prefers candidates in Noida or Pune, but welcomes talent from other Tier-1 tech hubs (Bangalore, Hyderabad, Mumbai, Delhi NCR, Chennai). Our `location_fit` feature scores Noida/Pune highest, and rewards candidates in other major hubs if they signal willingness to relocate (via the `willing_to_relocate` flag), mapping candidate mobility patterns to corporate needs.
3. **Assessment-Validated Competency:** Given the high volume of applicants in India, keyword stuffing is a common challenge. We treat Redrob's own platform `skill_assessment_scores` on AI/ML skills as a trust multiplier on top of candidate claims—ensuring that candidates who actually scored well on standardized assessments are prioritized over those who simply list tools.

### Pipeline — explicit two-stage retrieval


Like every real production ranking system, we separate **recall** (who is worth
ranking) from **precision** (what order). We made this split explicit after a
leave-one-out ablation showed that, within the top-100, `role_fit` is saturated
at 1.0 for everyone — it decides *entry*, not *order*. So `role_fit` is the gate;
the differentiators do the re-ranking.

```
candidates.jsonl
  └─ data.py                  stream + flatten to CandidateView (+ dense/sparse text)
  └─ features.py              named [0,1] features, each justified by a JD sentence
  └─ integrity.py             honeypot detection + JD disqualifier flags
  └─ retrieval.py             hybrid JD similarity: e5-small dense (cached) + BM25
  │
  ├─ STAGE 1  candidate_generation.py
  │     data-driven eligibility gate: non-honeypot AND (role_fit≥0.6 OR
  │     adjacent-title-with-strong-systems-evidence). 100k → ~10.9k, 100% recall
  │     of relevant candidates (measured, lossless).
  │
  └─ STAGE 2  rerank.py + ltr.py
        re-rank the eligible set on the DIFFERENTIATORS (experience, skill depth,
        systems evidence, semantic, location) — role_fit excluded (it's the gate).
        XGBoost LTR trained on the eligible set's JD-rubric proxy labels;
        interpretable linear blend kept as the defensible fallback.
  └─ scoring.py               base × behavioral-availability × disqualifier-penalty;
                              honeypot or ineligible → 0
  └─ reasoning.py             1-2 sentence justification from the facts that drove the rank
  └─ submission.py            write + self-validate the top-100 CSV
```

**Why weights aren't arbitrary.** The Stage-2 weights are *informed by* a
leave-one-out ablation (`scripts/derive_weights.py`), not hand-picked — each
differentiator's weight reflects its measured marginal contribution to the
composite, tempered by domain logic to avoid overfitting our own eval labels.
There is no "0.4 retrieval + 0.3 model + 0.3 LLM" guesswork, and crucially **no
LLM is called at rank time** (that would violate the no-network constraint and
fail Stage 3).

### The five things that matter, and how we handle each

| Challenge mechanic | Our response |
|---|---|
| **Keyword stuffers** (AI skills, wrong title) | `role_fit` (current/past title) gates the score; an explicit `keyword_stuffer` penalty (0.15x) when AI skills coexist with an off-role title and no ML in the career text. |
| **Honeypots** (~80, impossible profiles; >10% in top-100 = DQ) | A high-precision rule gate forces them to score 0. Calibrated on the full pool to catch the surgically-impossible profiles (expert skill with 0 months used; a role longer than the whole career) with no false positives. **0 honeypots in our top-100.** |
| **Plain-language Tier-5s** (real fits, no buzzwords) | Two-tier `system_evidence` rewards described work (recommendation/ranking/retrieval *and* production-ML/A-B-testing) from free text; dense e5 retrieval matches *intent*, surfacing them above keyword-stuffers. |
| **Behavioral twins** | A behavioral availability multiplier (recency, recruiter-response-rate, notice-period, open-to-work, interview reliability, verification) in `[0.55, 1.0]` re-orders otherwise-equal candidates without ever dominating fit. |
| **Reasoning quality** (Stage 4) | Reasoning is assembled only from facts that drove the score — never hallucinated — with tone keyed to rank. |

### Signal integration (the JD names specific signals — we use them)

The challenge asks to *"leverage all available information."* Beyond profile and
career text we use these `redrob_signals`, each tied to an explicit JD line:

- **`skill_assessment_scores`** → objective validation inside `must_have_skills`:
  a candidate who *lists* AI/ML skills but scored poorly on the platform's
  assessment of them is making hollow claims; a high score is validated ability.
  This is the strongest *objective* anti-keyword-stuffer signal in the data.
- **`recruiter_response_rate`, `last_active_date`, `interview_completion_rate`,
  `open_to_work_flag`, `notice_period_days`, verification flags** → the
  availability multiplier (JD: *"not actually available — down-weight"*;
  *"sub-30-day notice preferred"*).
- **`github_activity_score`** → lifts the bonus signal (JD values open-source;
  *"closed-source-only 5+ years"* is an anti-pattern).
- **`willing_to_relocate`** → location fit (Pune/Noida + relocation).

### Scoring, precisely

```
Stage 1:  eligible = (not honeypot) AND (role_fit ≥ 0.6
                       OR (adjacent role AND system_evidence ≥ 0.6))

Stage 2:  base  = Σ rerank_weight[f]·feature[f]  +  0.30·semantic_sim   ([0,1])
                  (or the XGBoost LTR prediction over the same differentiators)
          score = base · behavioral_mult · Π disqualifier_penalty
          score = 0   if honeypot or not eligible
```

Thresholds and the JD-justified feature definitions live in
[`jd_spec.py`](src/redrob_ranker/jd_spec.py); the gate is in
[`candidate_generation.py`](src/redrob_ranker/candidate_generation.py) and the
re-rank weights in [`rerank.py`](src/redrob_ranker/rerank.py). The interpretable
linear blend is a strong ranker on its own; the XGBoost layer learns non-linear
interactions on top of a **proxy relevance label distilled transparently from the
JD's own rubric** (we have no ground truth, so we encode the JD's stated "ideal
candidate" as the target — see [`ltr.py`](src/redrob_ranker/ltr.py)). The linear
score is retained on every candidate as an interpretable fallback and sanity check.

---

## Compute & reproducibility

* **CPU-only**, enforced in code (`device="cpu"`, `CUDA_VISIBLE_DEVICES=""`).
* **No network at rank time** — the embedding model loads from the local HF cache
  (`HF_HUB_OFFLINE=1`); nothing calls a hosted API.
* **Embeddings**: `intfloat/e5-small-v2`, 384-dim, ~130 MB, cached once.
  The 100k embedding matrix is ~150 MB float32 (well under 16 GB).
* The **ranking step** loads the cached matrix, embeds the single JD query, builds
  BM25 over the pool, trains the tiny LTR model, scores, and writes the CSV.

## Repo layout

```
rank.py                     # single reproduce command (Stage-3 entry point)
src/redrob_ranker/          # the package (one module per pipeline stage)
scripts/precompute.py       # offline embedding pre-computation
scripts/diagnose.py         # pool-level honeypot/disqualifier audit
scripts/inspect_candidate.py# per-candidate feature breakdown
tests/test_pipeline.py      # unit tests (pytest -q)
requirements.txt
Dockerfile                  # CPU-only reproduction container
submission_metadata.yaml    # portal metadata mirror
```

## Tests

```bash
pytest -q
```

## License / data

Candidate data is provided by the hackathon and is **not** committed
(`.gitignore`). Embeddings are regenerable via `scripts/precompute.py`.
