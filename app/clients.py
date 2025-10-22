# app/clients.py
from __future__ import annotations
import logging
import re
from functools import lru_cache
from realtime import Optional
from supabase import create_client, Client as SupabaseClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from app.config import SETTINGS
from app.vectorstores.supabase_retriever import SupabaseVectorStoreWrapper
from pydantic import Field

log = logging.getLogger("clients")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_URL_REGEX = re.compile(r"^(https?)://.+")


def _require_non_empty(name: str, value: Optional[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} não definido no ambiente.")
    return value.strip()


def _sanitize_supabase_url(url: str) -> str:
    url = url.rstrip("/")
    if not _URL_REGEX.match(url):
        raise RuntimeError("SUPABASE_URL inválida. Use http(s)://.")
    return url


@lru_cache(maxsize=1)
def get_supabase_client() -> SupabaseClient:
    supabase_url = _sanitize_supabase_url(_require_non_empty("SUPABASE_URL", getattr(SETTINGS, "SUPABASE_URL", None)))
    supabase_key = _require_non_empty("SUPABASE_SERVICE_ROLE_KEY", getattr(SETTINGS, "SUPABASE_SERVICE_ROLE_KEY", None))

    log.info("Criando cliente Supabase (cacheado).")
    try:
        # A lib já valida URL e chave, mas sanitizamos acima para evitar '//' e regex inválida.
        client = create_client(supabase_url, supabase_key)
    except Exception as e:
        # Mensagem clara para diagnóstico sem vazar segredos
        log.exception("Falha ao criar cliente Supabase: %s", e)
        raise
    return client


@lru_cache(maxsize=1)
def get_embedding() -> NVIDIAEmbeddings:
    if not SETTINGS.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY não definido no ambiente para NVIDIAEmbeddings.")
    emb = NVIDIAEmbeddings(model=SETTINGS.EMB_MODEL, api_key=SETTINGS.NVIDIA_API_KEY)
    log.info(f"Embeddings carregados: {SETTINGS.EMB_MODEL}")
    return emb


@lru_cache(maxsize=1)
def get_vectorstore() -> SupabaseVectorStoreWrapper:
    log.info(f"Criando VectorStore para tabela={SETTINGS.TABLE_NAME} query_fn={SETTINGS.QUERY_FN}")
    return SupabaseVectorStoreWrapper(
        client=get_supabase_client(),            
        embedding=get_embedding(),               
        table_name=SETTINGS.TABLE_NAME,
        query_name=SETTINGS.QUERY_FN,
    )