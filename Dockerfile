# syntax=docker/dockerfile:1

FROM python:3.13 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv
COPY pyproject.toml ./
# Instala dependências/projeto (assume que pyproject referencia seu pacote)
RUN .venv/bin/pip install .

FROM python:3.13-slim
WORKDIR /app

# Copia venv e código
COPY --from=builder /app/.venv .venv/
COPY . .

# Variáveis de ambiente - NÃO embute segredos!
# Segredos (deixar em branco; serão injetados via secrets):
ENV SUPABASE_SERVICE_ROLE_KEY=""
ENV NVIDIA_API_KEY=""

# Não-sigilosas (defaults seguros; sobrescreva via deploy)
ENV SUPABASE_URL=""
ENV TABLE_NAME="documents"
ENV CONTENT_COL="content"
ENV METADATA_COL="metadata"

ENV QUERY_FN="match_documents"
ENV FTS_QUERY_FN="match_documents_fts"
ENV USE_FTS="true"

ENV EMB_MODEL="nvidia/nv-embedqa-e5-v5"

ENV USE_RERANK="true"
ENV RERANK_TOP_N="10"
ENV NVIDIA_RERANK_MODEL="nvidia/nv-rerankqa-mistral-4b-v3"
# ENV NVIDIA_RERANK_BASE_URL=""

ENV LLM_MODEL="meta/llama-3.1-70b-instruct"
ENV TEMPERATURE="0.2"
ENV TOP_P="0.9"
ENV MAX_TOKENS="512"

ENV USE_MULTI_QUERY="true"
ENV K_SEM="8"
ENV K_KEYWORD="12"
ENV WEIGHT_SEM="0.65"
ENV WEIGHT_KEY="0.35"
ENV RRF_K="60"
ENV MAX_SNIPPET_CHARS="1200"

ENV TOKENIZER_NAME="intfloat/e5-large-v2"
ENV SUPABASE_PAGE_SIZE="1000"
ENV BM25_CACHE_TTL="600"

ENV CORS_ORIGIN="*"
ENV INGEST_FOLDER="data_pdfs"
ENV INGEST_BATCH_SIZE="100"

# Comando
CMD ["/app/.venv/bin/fastapi", "run"]
