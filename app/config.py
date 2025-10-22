# app/config.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # dotenv é opcional


@dataclass(frozen=True)
class Settings:
    # Supabase / Tabela
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    TABLE_NAME: str = os.getenv("TABLE_NAME", "documents")
    CONTENT_COL: str = os.getenv("CONTENT_COL", "content")
    METADATA_COL: str = os.getenv("METADATA_COL", "metadata")

    # RPCs
    QUERY_FN: str = os.getenv("QUERY_FN", "match_documents")
    FTS_QUERY_FN: str = os.getenv("FTS_QUERY_FN", "match_documents_fts")

    # Estratégias
    USE_FTS: bool = os.getenv("USE_FTS", "true").lower() in {"1", "true", "yes"}

    # Embeddings / NVIDIA
    EMB_MODEL: str = os.getenv("EMB_MODEL", "nvidia/nv-embedqa-e5-v5")
    NVIDIA_API_KEY: Optional[str] = os.getenv("NVIDIA_API_KEY")

    # Rerank NVIDIA
    USE_RERANK: bool = os.getenv("USE_RERANK", "false").lower() in {"1", "true", "yes"}
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
    NVIDIA_RERANK_MODEL: str = os.getenv("NVIDIA_RERANK_MODEL", "nvidia/nv-rerankqa-mistral-4b-v3")
    NVIDIA_RERANK_BASE_URL: Optional[str] = os.getenv("NVIDIA_RERANK_BASE_URL")

    # Paginação cache
    SUPABASE_PAGE_SIZE: int = int(os.getenv("SUPABASE_PAGE_SIZE", "1000"))
    BM25_CACHE_TTL: int = int(os.getenv("BM25_CACHE_TTL", "600"))

    # LLM Geração
    LLM_MODEL: str = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))

    # Multi-query
    USE_MULTI_QUERY: bool = os.getenv("USE_MULTI_QUERY", "true").lower() in {"1", "true", "yes"}

    # Hiperparâmetros híbrido
    K_SEM: int = int(os.getenv("K_SEM", "4"))
    K_KEYWORD: int = int(os.getenv("K_KEYWORD", "6"))
    WEIGHT_SEM: float = float(os.getenv("WEIGHT_SEM", "0.55"))
    WEIGHT_KEY: float = float(os.getenv("WEIGHT_KEY", "0.45"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # Contexto RAG
    MAX_SNIPPET_CHARS: int = int(os.getenv("MAX_SNIPPET_CHARS", "1200"))

    # Ingestão / Tokenizer
    TOKENIZER_NAME: str = os.getenv("TOKENIZER_NAME", "intfloat/e5-large-v2")

    def validate(self) -> None:
        missing = []
        if not self.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not self.SUPABASE_SERVICE_ROLE_KEY:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            raise RuntimeError(f"Variáveis de ambiente ausentes: {', '.join(missing)}. "
                               "Defina-as no .env ou no ambiente.")


SETTINGS = Settings()
SETTINGS.validate()