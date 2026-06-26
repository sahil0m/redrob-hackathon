# CPU-only reproduction container for the Redrob candidate ranker.
# Matches the Stage-3 sandbox constraints: CPU, no GPU, no network at rank time.
#
# Build:  docker build -t redrob-ranker .
# Run:    docker run --rm -v "$PWD:/work" redrob-ranker \
#             python rank.py --candidates /work/candidates.jsonl --out /work/submission.csv
#
# Pre-computation (one-time, may use network to fetch the embedding model once):
#         docker run --rm -v "$PWD:/work" redrob-ranker \
#             python scripts/precompute.py --candidates /work/candidates.jsonl --out-dir /work/artifacts

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# Install the CPU build of torch first, then the rest of the deps.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bring in the source, scripts, and any precomputed artifacts present at build time.
COPY src ./src
COPY scripts ./scripts
COPY rank.py ./rank.py
COPY artifacts ./artifacts

# The embedding model is fetched at precompute time and cached in the HF home;
# bake it into the image by running precompute once during the build if you want
# a fully offline image. By default we leave it to the precompute step.

ENTRYPOINT []
CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./submission.csv"]
