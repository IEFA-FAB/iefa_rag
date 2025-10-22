# app/vectorstores/supabase_retriever.py
from __future__ import annotations
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr, Field
from supabase import Client as SupabaseClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from lib.supabase_retriever import SupabaseVectorStore as LCSupabaseVectorStore


class SupabaseVectorStoreWrapper(BaseModel):
    """
    Wrapper fino sobre langchain_community.vectorstores.SupabaseVectorStore para manter
    compatibilidade com seu uso atual (método .as_retriever etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SupabaseClient = Field(repr=False)
    embedding: NVIDIAEmbeddings = Field(repr=False)
    table_name: str
    query_name: str
    content_column: str = "content"
    metadata_column: str = "metadata"

    _inner: LCSupabaseVectorStore = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        # Cria o vector store interno do LangChain com os campos já validados
        self._inner = LCSupabaseVectorStore(
            client=self.client,
            embedding=self.embedding,
            table_name=self.table_name,
            query_name=self.query_name,
        )

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: Optional[dict] = None
    ):
        return self._inner.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def add_documents(self, documents: list, ids: Optional[list[str]] = None, **kwargs: Any):
        return self._inner.add_documents(documents, ids=ids, **kwargs)