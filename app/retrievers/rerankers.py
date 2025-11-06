# app/retrievers/rerankers.py
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

import requests
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import SETTINGS

log = logging.getLogger("rerank")


class NvidiaRerankCompressor:
    """
    Compressor de documentos que utiliza o endpoint REST de reranking da NVIDIA.

    Parâmetros:
      - api_key: chave (NVIDIA NIM) para "Authorization: Bearer <API_KEY>"
      - model: ex. "nvidia/nv-rerankqa-mistral-4b-v3"
      - invoke_url: URL completa do endpoint; se None, é inferida a partir do modelo.
      - top_n: número máximo de passagens a manter após o reranking.
      - timeout: timeout de rede.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nvidia/nv-rerankqa-mistral-4b-v3",
        invoke_url: Optional[str] = None,
        top_n: int = 5,
        timeout: float = 20.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key é obrigatório para NvidiaRerankCompressor.")
        self.api_key = api_key
        self.model = model
        self.top_n = max(1, int(top_n))
        self.timeout = timeout
        self._session = session or requests.Session()
        # Aceita NVIDIA_RERANK_INVOKE_URL (preferencial) ou NVIDIA_RERANK_BASE_URL (se for URL completa)
        env_url = getattr(SETTINGS, "NVIDIA_RERANK_INVOKE_URL", None) or getattr(
            SETTINGS, "NVIDIA_RERANK_BASE_URL", None
        )
        self.invoke_url = invoke_url or env_url or self._infer_invoke_url(model)

        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        log.info(f"NvidiaRerankCompressor configurado (model={self.model}, top_n={self.top_n}).")
        log.debug(f"invoke_url={self.invoke_url}")

    @staticmethod
    def _infer_invoke_url(model: str) -> str:
        """
        Tenta inferir a URL a partir do nome do modelo no namespace 'nvidia/...'.

        Exemplo:
          model="nvidia/nv-rerankqa-mistral-4b-v3"
          -> https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking
        """
        base = "https://ai.api.nvidia.com/v1/retrieval"
        if "/" in model:
            provider, short = model.split("/", 1)
            return f"{base}/{provider}/{short}/reranking"
        return f"{base}/nvidia/{model}/reranking"

    def _build_payload(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        return {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": (d.page_content or "")} for d in docs],
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(
            self.invoke_url,
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_response(self, resp: Dict[str, Any], num_docs: int) -> List[int]:
        """
        Extrai índices ordenados por relevância. Tolerante a variações de schema.
        Retorna lista de índices (0-based) do mais relevante ao menos.
        """
        # Caso comum 1: {"results": [{"index": 0, "score": 0.93}, ...]}
        if isinstance(resp, dict) and "results" in resp and isinstance(resp["results"], list):
            items = resp["results"]
            ranked = []
            for it in items:
                idx = it.get("index")
                score = it.get("score")
                if isinstance(idx, int):
                    ranked.append((idx, float(score) if score is not None else 0.0))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in ranked]

        # Caso comum 2: {"data": [{"index": 0, "relevance_score": 0.93}, ...]}
        if "data" in resp and isinstance(resp["data"], list):
            items = resp["data"]
            ranked = []
            for it in items:
                idx = it.get("index")
                score = it.get("relevance_score") or it.get("score")
                if isinstance(idx, int):
                    ranked.append((idx, float(score) if score is not None else 0.0))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in ranked]

        # Fallback: "passages" com score
        if "passages" in resp and isinstance(resp["passages"], list):
            items = resp["passages"]
            ranked = []
            for i, it in enumerate(items):
                score = it.get("score") or it.get("relevance_score")
                idx = it.get("index", i)
                if isinstance(idx, int):
                    ranked.append((idx, float(score) if score is not None else 0.0))
            if ranked:
                ranked.sort(key=lambda x: x[1], reverse=True)
                return [idx for idx, _ in ranked]

        log.warning("Não foi possível interpretar a resposta do reranker. Mantendo ordem original.")
        return list(range(num_docs))

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Optional[Any] = None,  # mantido por compatibilidade de assinatura
    ) -> List[Document]:
        if not documents:
            return []

        payload = self._build_payload(query=query, docs=documents)
        try:
            resp = self._post(payload)
        except Exception as e:
            log.exception(f"Falha no reranking NVIDIA: {e}. Retornando documentos originais.")
            return documents

        order = self._parse_response(resp, num_docs=len(documents))

        # Dedup e somente índices válidos
        seen = set()
        order = [i for i in order if isinstance(i, int) and 0 <= i < len(documents) and not (i in seen or seen.add(i))]

        # Aplica top_n
        order = order[: self.top_n]
        return [documents[i] for i in order]


def maybe_wrap_with_reranker(base_retriever: BaseRetriever, top_n: Optional[int] = None) -> BaseRetriever:
    if not SETTINGS.USE_RERANK:
        log.info("Rerank desabilitado via configuração (USE_RERANK=False). Seguindo sem rerank.")
        return base_retriever

    effective_top_n = int(top_n or getattr(SETTINGS, "RERANK_TOP_N", 5) or 5)

    if not (getattr(SETTINGS, "NVIDIA_API_KEY", None) and getattr(SETTINGS, "NVIDIA_RERANK_MODEL", None)):
        log.warning("Config NVIDIA incompleta (API key ou modelo ausentes). Seguindo sem rerank.")
        return base_retriever

    try:
        reranker = NvidiaRerankCompressor(
            api_key=SETTINGS.NVIDIA_API_KEY,
            model=SETTINGS.NVIDIA_RERANK_MODEL,  # ex.: "nvidia/nv-rerankqa-mistral-4b-v3"
            # Se preferir, defina explicitamente em SETTINGS.NVIDIA_RERANK_INVOKE_URL
            invoke_url=getattr(SETTINGS, "NVIDIA_RERANK_INVOKE_URL", None)
            or getattr(SETTINGS, "NVIDIA_RERANK_BASE_URL", None),
            top_n=effective_top_n,
            timeout=float(getattr(SETTINGS, "NVIDIA_RERANK_TIMEOUT", 20.0)),
        )

        wrapped = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=reranker,  # qualquer objeto com compress_documents serve
        )
        log.info(
            f"Rerank NVIDIA habilitado (modelo={SETTINGS.NVIDIA_RERANK_MODEL}, top_n={effective_top_n})."
        )
        return wrapped

    except Exception as e:
        log.exception(f"Falha ao habilitar rerank NVIDIA: {e}. Seguindo sem rerank.")
        return base_retriever