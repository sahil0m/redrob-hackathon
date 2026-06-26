"""
jd_spec.py — The job description, encoded as structured, auditable fit criteria.

This module is the single source of truth for *what the JD means*. Every other
scoring decision in the ranker traces back to a constant defined here, and every
constant carries a `why` string quoting the JD sentence it comes from. That
traceability is deliberate: at Stage 5 (defend-your-work interview) we want to be
able to point at any weight and name the JD line that justifies it, and at Stage 4
the reasoning we emit must cite only facts that actually drove the score.

The JD is "Senior AI Engineer — Founding Team" at Redrob AI. The full text lives
in the bundle (job_description.md). We do NOT keyword-match against it; we model
the *intent* the JD spells out in its "How to read between the lines" and
"Things we explicitly do NOT want" sections.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# 1. The "ideal candidate" envelope (JD: "How to read between the lines")
# --------------------------------------------------------------------------- #
# "6-8 years total experience, of which 4-5 are in applied ML/AI roles at
#  product companies (not pure services)."
IDEAL_YOE_LOW = 6.0
IDEAL_YOE_HIGH = 8.0
# "Experience Required: 5-9 years ... This is a range, not a requirement ...
#  we'll seriously consider candidates outside the band if other signals are
#  strong." -> soft band, not a hard gate.
ACCEPTABLE_YOE_LOW = 4.0   # "Some people hit senior judgment at 4 years"
ACCEPTABLE_YOE_HIGH = 12.0  # taper above; >12 increasingly unlikely for "senior IC who codes"


# --------------------------------------------------------------------------- #
# 2. Role / domain relevance (JD: the role is retrieval, ranking, matching, LLMs)
# --------------------------------------------------------------------------- #
# Titles that signal the candidate *does the work this role does*. These are not
# matched as keywords against the skills list — they are matched against
# current_title and the title history, where they actually mean something.
CORE_TITLE_TERMS = (
    "machine learning engineer", "ml engineer", "applied ml", "applied scientist",
    "ai engineer", "ai research engineer", "research engineer",
    "data scientist", "nlp engineer", "search engineer", "ranking engineer",
    "recommendation", "relevance engineer",
)
# Adjacent titles: a plausible path into the role (JD explicitly blesses the
# Tier-5 data/backend engineer "who built a recommendation system at a product
# company"). Rewarded, but less than core.
ADJACENT_TITLE_TERMS = (
    "data engineer", "analytics engineer", "backend engineer", "software engineer",
    "full stack", "platform engineer", "mlops",
)

# Evidence of having *built the systems the JD cares about*, found in free-text
# career descriptions and summary. This is how we reward plain-language Tier-5s
# who never write "RAG" but describe building a recommender at scale.
# JD: "if their career history shows they built a recommendation system at a
#      product company, they're a fit."
SYSTEM_EVIDENCE_TERMS = (
    "recommendation", "recommender", "ranking", "search relevance", "retrieval",
    "semantic search", "vector search", "embedding", "personalization",
    "information retrieval", "learning to rank", "matching system", "feed ranking",
)
# Supporting evidence: real applied-ML / production / evaluation work that the JD
# values ("shipped to real users", "production deployment", "A/B testing") but
# that isn't itself a ranking/retrieval system. Calibrated against the pool:
# "data pipeline" (~4.3k), "production" (~2.8k), "ml model" (~1.5k),
# "a/b test" (~1.4k) are common among relevant candidates, so they count as
# *supporting* signal at lower weight, never as the bullseye.
SUPPORTING_EVIDENCE_TERMS = (
    "production", "deployed", "real users", "at scale", "ml model", "machine learning",
    "model training", "a/b test", "ab test", "predictive model", "feature pipeline",
    "data pipeline", "deep learning", "neural network", "nlp", "scoring model",
)
# JD "Things you absolutely need": embeddings retrieval, vector DBs / hybrid
# search, evaluation frameworks for ranking.
RETRIEVAL_TECH_TERMS = (
    "embedding", "sentence-transformer", "bge", "e5", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "bm25",
    "hybrid search", "vector database", "vector db", "ann ",
)
EVAL_FRAMEWORK_TERMS = (
    "ndcg", "mrr", "map@", "mean average precision", "a/b test", "ab test",
    "offline evaluation", "online evaluation", "ranking metric", "precision@",
    "recall@", "offline-to-online",
)
LLM_TERMS = (
    "llm", "fine-tun", "lora", "qlora", "peft", "rag", "retrieval-augmented",
    "prompt", "large language model", "transformer",
)


# --------------------------------------------------------------------------- #
# 3. Explicit disqualifiers (JD: "the disqualifiers we actually apply" +
#    "Things we explicitly do NOT want"). These are PENALTIES, not all hard
#    gates — the JD hedges most with "probably". We model severity accordingly.
# --------------------------------------------------------------------------- #

# "People who have only worked at consulting firms ... in their entire career."
# Note the carve-out: "If you're currently at one of these companies but have
# prior product-company experience, that's fine." -> only penalize if the WHOLE
# career is services.
CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
    "deloitte", "ibm services",
)

# "Title-chasers ... switching companies every 1.5 years ... We need someone who
#  plans to be here for 3+ years." -> short median tenure across distinct
#  employers is the signal.
TITLE_CHASER_MAX_MEDIAN_TENURE_MONTHS = 18.0
TITLE_CHASER_MIN_JOBS = 4  # need enough hops for the pattern to be real

# "primary expertise is computer vision, speech, or robotics without significant
#  NLP/IR exposure." -> penalize CV/speech-dominated profiles lacking NLP/IR.
OTHER_DOMAIN_TERMS = (
    "computer vision", "image classification", "object detection", "speech",
    "robotics", "tts", "asr", "gans", "segmentation", "ocr", "pose estimation",
)
NLP_IR_TERMS = (
    "nlp", "natural language", "information retrieval", "search", "ranking",
    "retrieval", "text", "embedding", "transformer", "bert", "language model",
)


# --------------------------------------------------------------------------- #
# 4. Location (JD: Pune/Noida preferred, relocation-friendly Tier-1 cities)
# --------------------------------------------------------------------------- #
# "Located in or willing to relocate to Noida or Pune."
# "Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome."
PREFERRED_CITIES = ("noida", "pune")
TIER1_INDIAN_CITIES = (
    "noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "chennai", "ncr",
)
# "Outside India: case-by-case, but we don't sponsor work visas." -> non-India,
# non-relocating candidates are heavily down-weighted but not zeroed.


# --------------------------------------------------------------------------- #
# 5. Behavioral availability (JD final note + redrob_signals_doc)
# --------------------------------------------------------------------------- #
# "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5%
#  recruiter response rate is, for hiring purposes, not actually available.
#  Down-weight them appropriately." -> a MULTIPLIER on the fit score, never the
#  primary driver (a great-but-quiet candidate should still outrank a
#  mediocre-but-chatty one).
REFERENCE_DATE = "2026-06-01"  # anchor for recency (dataset is May/June 2026)
STALE_DAYS = 180               # ~6 months, the JD's own example
LOW_RESPONSE_RATE = 0.05       # the JD's own "5%" example


# NOTE on scoring weights: the per-feature scoring weights live in rerank.py
# (Stage 2), NOT here. They are split into a Stage-1 gate (role_fit) and Stage-2
# re-rank differentiators, and the re-rank weights are informed by a leave-one-out
# ablation rather than hand-set. See rerank.py and scripts/derive_weights.py.

# The JD text, distilled into a single string we embed once for dense retrieval.
# This is the "query" side of the semantic match. It is intentionally written in
# the JD's own framing (product over research, systems over frameworks) so that
# the embedding captures intent, not just the buzzword surface.
JD_QUERY_TEXT = (
    "Senior AI Engineer for a Series A AI-native talent intelligence product company. "
    "Owns the intelligence layer: ranking, retrieval, and candidate-job matching systems. "
    "Needs production experience with embeddings-based retrieval (sentence-transformers, BGE, E5), "
    "vector databases and hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch), "
    "and rigorous evaluation of ranking systems (NDCG, MRR, MAP, A/B testing, offline-to-online correlation). "
    "Strong Python and code quality. Has shipped an end-to-end ranking, search, or recommendation system "
    "to real users at meaningful scale at a product company, not a pure services or consulting firm. "
    "Scrappy product engineering attitude, ships fast, still writes production code. "
    "Six to eight years total experience, four to five in applied machine learning at product companies. "
    "Strong opinions on hybrid versus dense retrieval, when to fine-tune versus prompt LLMs. "
    "Not a title-chaser, not a pure researcher, not LLM-API-only via LangChain, not computer vision or "
    "speech or robotics without NLP and information retrieval. Located in or willing to relocate to Pune or Noida."
)
