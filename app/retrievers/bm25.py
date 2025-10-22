# app/retrievers/bm25.py
from __future__ import annotations
import time
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.config import SETTINGS
from app.clients import get_supabase_client
from app.utils.text import tokenize_pt, coerce_metadata
from supabase import Client as SupabaseClient
from pydantic import Field

log = logging.getLogger("bm25")

_BM25_CACHE = {
    "retriever": None,
    "expires_at": 0.0,
    "doc_count": 0,
}


def _fetch_all_rows_from_supabase() -> List[Dict[str, Any]]:
    client: SupabaseClient = Field(default_factory=get_supabase_client, repr=False)
    rows: List[Dict[str, Any]] = []
    page_size = SETTINGS.SUPABASE_PAGE_SIZE
    start = 0

    log.info(f"Carregando linhas do Supabase em páginas de {page_size}...")
    while True:
        end = start + page_size - 1
        resp = (
            client.table(SETTINGS.TABLE_NAME)
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


def get_bm25_retriever(k: int = 5) -> BM25Retriever:
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
        metadata = coerce_metadata(r.get(SETTINGS.METADATA_COL))
        docs.append(Document(page_content=content, metadata=metadata))

    if not docs:
        log.warning("Nenhum documento com conteúdo encontrado para BM25.")
        retr = BM25Retriever.from_documents([], preprocess_func=tokenize_pt)
    else:
        log.info(f"Construindo índice BM25 em {len(docs)} documentos...")
        retr = BM25Retriever.from_documents(docs, preprocess_func=tokenize_pt)
    retr.k = k

    _BM25_CACHE["retriever"] = retr
    _BM25_CACHE["expires_at"] = now + SETTINGS.BM25_CACHE_TTL
    _BM25_CACHE["doc_count"] = len(docs)
    return retr