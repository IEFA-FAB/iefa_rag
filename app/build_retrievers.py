# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Optional
from unidecode import unidecode

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # dotenv é opcional

from pydantic import Field, ConfigDict

from supabase import create_client, Client as SupabaseClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from lib.supabase_retriever import SupabaseVectorStore  # seu wrapper para pgvector
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import CohereRerank

# ------------------------------------------------------------------------------
# Log
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("retriever")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    TABLE_NAME: str = os.getenv("TABLE_NAME", "documents")

    # RPC para busca SEMÂNTICA (pgvector)
    QUERY_FN: str = os.getenv("QUERY_FN", "match_documents")

    # RPC para FTS (texto)
    FTS_QUERY_FN: str = os.getenv("FTS_QUERY_FN", "match_documents_fts")

    # Toggle: usar FTS em vez de BM25 no híbrido
    USE_FTS: bool = os.getenv("USE_FTS", "true").lower() in {"1", "true", "yes"}

    EMB_MODEL: str = os.getenv("EMB_MODEL", "nvidia/nv-embedqa-e5-v5")
    CONTENT_COL: str = os.getenv("CONTENT_COL", "content")
    METADATA_COL: str = os.getenv("METADATA_COL", "metadata")
    NVIDIA_API_KEY: Optional[str] = os.getenv("NVIDIA_API_KEY")  # requerido no runtime de embeddings

    # Ajustes de paginação para Supabase
    SUPABASE_PAGE_SIZE: int = int(os.getenv("SUPABASE_PAGE_SIZE", "1000"))

    # Cache do BM25 (em segundos)
    BM25_CACHE_TTL: int = int(os.getenv("BM25_CACHE_TTL", "600"))  # 10 minutos

    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    USE_RERANK: bool = os.getenv("USE_RERANK", "false").lower() in {"1", "true", "yes"}
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))


    def validate(self) -> None:
        missing = []
        if not self.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not self.SUPABASE_SERVICE_ROLE_KEY:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            raise RuntimeError(f"Variáveis de ambiente ausentes: {', '.join(missing)}. "
                               f"Defina-as no .env ou no ambiente.")
        # NVIDIA_API_KEY validada apenas na criação dos embeddings

SETTINGS = Settings()
SETTINGS.validate()

# ------------------------------------------------------------------------------
# Clientes e recursos com cache
# ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_supabase_client() -> SupabaseClient:
    log.info("Criando cliente Supabase (cacheado).")
    return create_client(SETTINGS.SUPABASE_URL, SETTINGS.SUPABASE_SERVICE_ROLE_KEY)

@lru_cache(maxsize=1)
def get_embedding() -> NVIDIAEmbeddings:
    """
    Cria embeddings da NVIDIA (cacheados). Requer NVIDIA_API_KEY no ambiente.
    """
    if not SETTINGS.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY não definido no ambiente para NVIDIAEmbeddings.")
    emb = NVIDIAEmbeddings(
        model=SETTINGS.EMB_MODEL,
        api_key=SETTINGS.NVIDIA_API_KEY,
        # timeout=60,
        # max_retries=3,
    )
    log.info(f"Embeddings carregados: {SETTINGS.EMB_MODEL}")
    return emb

@lru_cache(maxsize=1)
def get_vectorstore() -> SupabaseVectorStore:
    supabase = get_supabase_client()
    emb = get_embedding()
    log.info(f"Criando VectorStore para tabela={SETTINGS.TABLE_NAME} query_fn={SETTINGS.QUERY_FN}")
    return SupabaseVectorStore(
        client=supabase,
        embedding=emb,
        table_name=SETTINGS.TABLE_NAME,
        query_name=SETTINGS.QUERY_FN,
    )

# ------------------------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------------------------
def _normalize_text_pt(text: str) -> str:
    s = text or ""
    s = s.strip().lower()
    try:
        s = unidecode(s)
    except Exception:
        pass
    s = " ".join(s.split())
    return s

def _tokenize_pt(text: str) -> list[str]:
    s = _normalize_text_pt(text)
    return s.split()

def _fetch_all_rows_from_supabase() -> List[Dict[str, Any]]:
    supabase = get_supabase_client()
    rows: List[Dict[str, Any]] = []
    page_size = SETTINGS.SUPABASE_PAGE_SIZE
    start = 0

    log.info(f"Carregando linhas do Supabase em páginas de {page_size}...")
    while True:
        end = start + page_size - 1  # inclusive
        resp = (
            supabase.table(SETTINGS.TABLE_NAME)
            .select(f"{SETTINGS.CONTENT_COL},{SETTINGS.METADATA_COL}")
            .range(start, end)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        log.info(f"- Recebidas {len(batch)} linhas (total acumulado: {len(rows)})")
        if len(batch) < page_size:
            break
        start += page_size
    return rows

def _coerce_metadata(meta: Any) -> Dict[str, Any]:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str) and meta:
        try:
            return json.loads(meta)
        except Exception:
            return {"raw_metadata": meta}
    return {}

def maybe_wrap_with_reranker(
    base_retriever: BaseRetriever,
    top_n: Optional[int] = None
) -> BaseRetriever:
    """
    Se USE_RERANK estiver habilitado e houver COHERE_API_KEY, envolve o retriever base
    com um ContextualCompressionRetriever usando Cohere Rerank.
    Caso contrário, retorna o retriever base inalterado.
    """
    if not SETTINGS.USE_RERANK:
        return base_retriever

    if not SETTINGS.COHERE_API_KEY:
        log.warning("USE_RERANK=True, mas COHERE_API_KEY não está definida. Seguindo sem rerank.")
        return base_retriever

    try:
        compressor = CohereRerank(
            model="rerank-3.5",
            top_n=top_n or SETTINGS.RERANK_TOP_N,
            cohere_api_key=SETTINGS.COHERE_API_KEY,
        )
        reranked = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            document_compressor=compressor,
        )
        log.info(f"Rerank habilitado (Cohere Rerank, top_n={top_n or SETTINGS.RERANK_TOP_N}).")
        return reranked
    except Exception as e:
        log.exception(f"Falha ao habilitar rerank: {e}. Seguindo sem rerank.")
        return base_retriever

# ------------------------------------------------------------------------------
# Retrievers
# ------------------------------------------------------------------------------
def get_semantic_retriever(k: int = 5):
    """
    Retriever semântico (pgvector no Supabase via seu VectorStore).
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": k,"fetch_k": 50, "lambda_mult": 0.3})

# Cache simples em memória para BM25
_BM25_CACHE = {
    "retriever": None,
    "expires_at": 0.0,
    "doc_count": 0,
}

def get_bm25_retriever(k: int = 5) -> BM25Retriever:
    """
    BM25 em memória (bom para bases pequenas/médias).
    """
    now = time.time()
    cached = _BM25_CACHE.get("retriever")
    if cached and _BM25_CACHE["expires_at"] > now:
        log.info("Reutilizando BM25 do cache.")
        retr: BM25Retriever = cached
        retr.k = k
        return retr

    rows = _fetch_all_rows_from_supabase()
    docs: List[Document] = []
    for r in rows:
        content = r.get(SETTINGS.CONTENT_COL) or ""
        if not content:
            continue
        metadata = _coerce_metadata(r.get(SETTINGS.METADATA_COL))
        docs.append(Document(page_content=content, metadata=metadata))

    if not docs:
        log.warning("Nenhum documento com conteúdo encontrado para BM25.")
        retr = BM25Retriever.from_documents([], preprocess_func=_tokenize_pt)
    else:
        log.info(f"Construindo índice BM25 em {len(docs)} documentos...")
        retr = BM25Retriever.from_documents(docs, preprocess_func=_tokenize_pt)
    retr.k = k

    _BM25_CACHE["retriever"] = retr
    _BM25_CACHE["expires_at"] = now + SETTINGS.BM25_CACHE_TTL
    _BM25_CACHE["doc_count"] = len(docs)
    return retr

# --- NOVO: FTS Retriever ------------------------------------------------------
class SupabaseFTSRetriever(BaseRetriever):
    """
    Retriever baseado em FTS no Postgres via RPC Supabase.

    Pré-requisito: função RPC (default: match_documents_fts) com assinatura:
      match_documents_fts(query_text text, match_count int, filter jsonb)
    retornando colunas: id, content, metadata, rank, highlights
    """
    client: SupabaseClient
    rpc_name: str
    k: int = 10
    filters: Dict[str, Any] = Field(default_factory=dict)

    # Permite tipos arbitrários (ex.: cliente Supabase)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        payload = {
            "query_text": query,
            "match_count": self.k,
            "filter": self.filters or {},
        }
        res = self.client.rpc(self.rpc_name, payload).execute()
        rows = res.data or []
        docs: List[Document] = []
        for r in rows:
            content = r.get("content") or ""
            metadata = _coerce_metadata(r.get("metadata"))
            # Adiciona rank/highlights no metadata
            metadata["rank"] = r.get("rank")
            if r.get("highlights"):
                metadata["highlights"] = r.get("highlights")
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

def get_fts_retriever(k: int = 10, filters: Optional[dict] = None) -> SupabaseFTSRetriever:
    """
    Constrói um retriever FTS chamando a RPC configurada em FTS_QUERY_FN.
    """
    supabase = get_supabase_client()
    rpc = SETTINGS.FTS_QUERY_FN
    log.info(f"Criando FTS retriever via RPC={rpc} (k={k})")
    return SupabaseFTSRetriever(client=supabase, rpc_name=rpc, k=k, filters=filters or {})

def get_hybrid_retriever(
    k_sem: int = 4,
    k_keyword: int = 6,
    weights: tuple[float, float] = (0.5, 0.5),
    rrf_k: int = 60,
    filters: Optional[dict] = None,
) -> EnsembleRetriever:
    """
    Retriever híbrido:
      - Se USE_FTS=True: FTS + Semântico
      - Caso contrário: BM25 + Semântico
    """
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

# ------------------------------------------------------------------------------
# Execução de teste local
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    query = "Qual é o procedimento descrito no documento X?"
    # Ajuste USE_FTS=true/false no .env para alternar
    hybrid = get_hybrid_retriever(k_sem=4, k_keyword=6, weights=(0.5, 0.5), rrf_k=60)
    docs = hybrid.get_relevant_documents(query)

    print("\nTop documentos recuperados (híbrido):")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source") or d.metadata.get("id") or "<sem fonte>"
        hi = d.metadata.get("highlights")
        snippet = (hi or d.page_content or "")[:200].replace("\n", " ")
        rank = d.metadata.get("rank")
        print(f"{i:02d} - {src} | rank={rank}: {snippet}...")