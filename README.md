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

## Why this design

The challenge is explicitly built to punish keyword matching. The provided
`sample_submission.csv` is itself the trap — it ranks HR Managers and Graphic
Designers at the top because they *list* AI skills. The JD tells participants
directly: a Marketing Manager with every AI keyword is **not** a fit; a data
engineer who *built a recommender at a product company* **is**, even without the
buzzwords. So every design choice here optimises for **reading the profile**, not
the vocabulary.

### Pipeline

```
candidates.jsonl
  └─ data.py          stream + flatten to CandidateView (+ dense/sparse text)
  └─ integrity.py     honeypot gate (hard) + JD disqualifier flags (penalties)
  └─ features.py      named [0,1] features, each justified by a JD sentence
  └─ retrieval.py     hybrid JD similarity: e5-small dense (cached) + BM25
  └─ ltr.py           XGBoost over the features, trained on a JD-rubric proxy label
  └─ scoring.py       base x behavioral-availability x disqualifier-penalty; honeypot→0
  └─ reasoning.py     1-2 sentence justification from the facts that drove the score
  └─ submission.py    write + self-validate the top-100 CSV
```

### The five things that matter, and how we handle each

| Challenge mechanic | Our response |
|---|---|
| **Keyword stuffers** (AI skills, wrong title) | `role_fit` (current/past title) gates the score; an explicit `keyword_stuffer` penalty (0.15x) when AI skills coexist with an off-role title and no ML in the career text. |
| **Honeypots** (~80, impossible profiles; >10% in top-100 = DQ) | A high-precision rule gate forces them to score 0. Calibrated on the full pool to catch the surgically-impossible profiles (expert skill with 0 months used; a role longer than the whole career) with no false positives. **0 honeypots in our top-100.** |
| **Plain-language Tier-5s** (real fits, no buzzwords) | Two-tier `system_evidence` rewards described work (recommendation/ranking/retrieval *and* production-ML/A-B-testing) from free text; dense e5 retrieval matches *intent*, surfacing them above keyword-stuffers. |
| **Behavioral twins** | A behavioral availability multiplier (recency, recruiter-response-rate, open-to-work, verification, interview reliability) in `[0.55, 1.0]` re-orders otherwise-equal candidates without ever dominating fit. |
| **Reasoning quality** (Stage 4) | Reasoning is assembled only from facts that drove the score — never hallucinated — with tone keyed to rank. |

### Scoring, precisely

```
base    = Σ weight[f]·feature[f]  +  0.35·semantic_sim     (renormalised to [0,1])
        (or the XGBoost LTR prediction over the same features)
score   = base · behavioral_mult · Π disqualifier_penalty
score   = 0   if honeypot
```

Every `weight` and threshold lives in [`jd_spec.py`](src/redrob_ranker/jd_spec.py)
with a comment quoting the JD sentence that justifies it. The linear blend is a
strong, fully-interpretable ranker on its own; the XGBoost layer learns the
non-linear feature interactions on top of a **proxy relevance label distilled
transparently from the JD's own rubric** (we have no access to ground truth, so we
encode the JD's stated "ideal candidate" as the target — see
[`ltr.py`](src/redrob_ranker/ltr.py)). The linear score is retained on every
candidate as an interpretable fallback and sanity check.

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
