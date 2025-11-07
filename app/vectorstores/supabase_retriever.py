# app/vectorstores/supabase_retriever.py
from __future__ import annotations
import os
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, ConfigDict, PrivateAttr, Field
from supabase import Client as SupabaseClient
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from lib.supabase_retriever import SupabaseVectorStore as LCSupabaseVectorStore


class NvidiaOpenAIEmbeddings(Embeddings):
    """
    Implementação de Embeddings compatível com LangChain que usa o SDK 'openai'
    apontando para o endpoint OpenAI-compatível da NVIDIA.

    Por padrão, usa o modelo 'baai/bge-m3' com encoding 'float' e truncate 'NONE'.
    """

    def __init__(
        self,
        model: str = "baai/bge-m3",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        encoding_format: str = "float",
        truncate: str = "END",
        client: Optional[OpenAI] = None,
        max_retries: int = 5,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.encoding_format = encoding_format
        self.truncate = truncate

        # Prefira NVIDIA_API_KEY (evite reutilizar OPENAI_API_KEY para não confundir)
        resolved_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not resolved_key:
            # fallback opcional: OPENAI_API_KEY
            resolved_key = os.getenv("OPENAI_API_KEY")

        self._client = client or OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

    def _extra_body(self) -> Dict[str, str]:
        # Campo específico da NVIDIA para embeddings (não faz parte do spec OpenAI)
        return {"truncate": self.truncate} if self.truncate else {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format=self.encoding_format,
            extra_body=self._extra_body(),
        )
        # A API mantém a ordem das entradas.
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(
            input=text,  # pode ser string ou lista; string funciona bem para 1 item
            model=self.model,
            encoding_format=self.encoding_format,
            extra_body=self._extra_body(),
        )
        return resp.data[0].embedding


class SupabaseVectorStoreWrapper(BaseModel):
    """
    Wrapper fino sobre langchain_community.vectorstores.SupabaseVectorStore para manter
    compatibilidade com seu uso atual (método .as_retriever etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: SupabaseClient = Field(repr=False)
    embedding: Embeddings = Field(repr=False)
    table_name: str
    query_name: str
    content_column: str = "content"
    metadata_column: str = "metadata"

    _inner: LCSupabaseVectorStore = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
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