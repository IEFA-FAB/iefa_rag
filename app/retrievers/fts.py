# app/retrievers/fts.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from app.clients import get_supabase_client
from app.config import SETTINGS
from app.utils.text import coerce_metadata
from supabase import Client as SupabaseClient


class SupabaseFTSRetriever(BaseRetriever):
    """
    Retriever baseado em FTS no Postgres via RPC Supabase.

    Pré-requisito: RPC (FTS_QUERY_FN) com assinatura:
      match_documents_fts(query_text text, match_count int, filter jsonb)
    retornando colunas: id, content, metadata, rank, highlights
    """
    client: SupabaseClient = Field(default_factory=get_supabase_client, repr=False)
    rpc_name: str = SETTINGS.FTS_QUERY_FN
    k: int = 10
    filters: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        payload = {"query_text": query, "match_count": self.k, "filter": self.filters or {}}
        res = self.client.rpc(self.rpc_name, payload).execute()
        rows = res.data or []
        docs: List[Document] = []
        for r in rows:
            content = r.get("content") or ""
            metadata = coerce_metadata(r.get("metadata"))
            metadata["rank"] = r.get("rank")
            if r.get("highlights"):
                metadata["highlights"] = r.get("highlights")
            docs.append(Document(page_content=content, metadata=metadata))
        return docs


def get_fts_retriever(k: int = 10, filters: Optional[dict] = None) -> SupabaseFTSRetriever:
    retr = SupabaseFTSRetriever()
    retr.k = k
    retr.filters = filters or {}
    return retr