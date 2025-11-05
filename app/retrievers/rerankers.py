# app/retrievers/rerankers.py
from __future__ import annotations
import logging
from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_nvidia_ai_endpoints import NVIDIARerank
from app.config import SETTINGS

log = logging.getLogger("rerank")


def maybe_wrap_with_reranker(base_retriever: BaseRetriever, top_n: Optional[int] = None) -> BaseRetriever:
    if not SETTINGS.USE_RERANK:
        log.info("Rerank desabilitado via configuração (USE_RERANK=False). Seguindo sem rerank.")
        return base_retriever

    # top_n efetivo
    effective_top_n = top_n or SETTINGS.RERANK_TOP_N

    # pré-validações simples de config para evitar exceções caras
    if not (SETTINGS.NVIDIA_API_KEY and SETTINGS.NVIDIA_RERANK_MODEL):
        log.warning("Config NVIDIA incompleta (API key ou modelo ausentes). Seguindo sem rerank.")
        return base_retriever

    try:
        reranker = NVIDIARerank(
            model=SETTINGS.NVIDIA_RERANK_MODEL,
            api_key=SETTINGS.NVIDIA_API_KEY,
            base_url=SETTINGS.NVIDIA_RERANK_BASE_URL or None,
            top_n=effective_top_n,
            truncate="END",
        )
        wrapped = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=reranker,  # <- correção aqui
        )
        log.info(
            f"Rerank NVIDIA habilitado (modelo={SETTINGS.NVIDIA_RERANK_MODEL}, top_n={effective_top_n})."
        )
        return wrapped

    except Exception as e:
        # Mantém o fluxo funcionando mesmo se o rerank falhar
        log.exception(f"Falha ao habilitar rerank NVIDIA: {e}. Seguindo sem rerank.")
        return base_retriever