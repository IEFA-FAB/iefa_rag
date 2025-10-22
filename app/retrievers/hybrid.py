# app/retrievers/hybrid.py
from __future__ import annotations
import logging
from typing import Optional
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from app.retrievers.semantic import get_semantic_retriever
from app.retrievers.bm25 import get_bm25_retriever
from app.retrievers.fts import get_fts_retriever
from app.retrievers.rerankers import maybe_wrap_with_reranker
from app.config import SETTINGS

log = logging.getLogger("hybrid")


def get_hybrid_retriever(
    k_sem: int = 4,
    k_keyword: int = 6,
    weights: tuple[float, float] = (0.5, 0.5),
    rrf_k: int = 60,
    filters: Optional[dict] = None,
) -> BaseRetriever:
    sem = get_semantic_retriever(k=k_sem)
    if SETTINGS.USE_FTS:
        keyword = get_fts_retriever(k=k_keyword, filters=filters)
        log.info("Híbrido: FTS + Semântico")
    else:
        keyword = get_bm25_retriever(k=k_keyword)
        log.info("Híbrido: BM25 + Semântico")

    hybrid = EnsembleRetriever(
        retrievers=[keyword, sem],
        weights=list(weights),
        rrf_k=rrf_k,
    )
    return maybe_wrap_with_reranker(hybrid, top_n=None)