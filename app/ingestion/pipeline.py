
from dataclasses import dataclass
import os
import re
import glob
import uuid
import math
from typing import List, Iterable, Optional
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore as LCSupabaseVectorStore
from langchain.schema import Document
""" from transformers import AutoTokenizer """
from openai import OpenAI  # OpenAI SDK falando com a API da NVIDIA
import hashlib
import time


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

# =========================
# Utils
# =========================
def batched(iterable: List, n: int) -> Iterable[List]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n', '', text)     # junta hifen + quebra de linha
    text = re.sub(r'\s*\n\s*', '\n', text) # normaliza quebras
    text = re.sub(r'\s+', ' ', text)       # comprime espaços
    return text.strip()


def dedup_chunks(chunks: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for c in chunks:
        key = hashlib.md5(c.page_content.encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def load_pdfs_from_folder(folder: str) -> List[Document]:
    docs = []
    pdf_paths = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            for d in pages:
                d.metadata = d.metadata or {}
                d.metadata.update({
                    "source": os.path.basename(path),
                    "path": os.path.abspath(path),
                })
                docs.append(d)
        except Exception as e:
            print(f"[WARN] Falha ao ler {path}: {e}")
    return docs


""" def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
 """

""" def validate_chunks(
    chunks: List[Document],
    tokenizer_name: str,
    max_tokens: int = 512,
    show_examples: int = 5,
) -> None:
    tok = get_tokenizer(tokenizer_name)
    too_long = []
    for i, c in enumerate(chunks):
        n = len(tok(c.page_content, add_special_tokens=False)["input_ids"])
        if n > max_tokens:
            too_long.append((i, n))
    print(f"Chunks acima de {max_tokens} tokens: {len(too_long)}")
    if too_long[:show_examples]:
        print("Exemplos:", too_long[:show_examples]) """


# =========================
# bge-m3 Embeddings (via NVIDIA OpenAI API)
# =========================
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class NvidiaOpenAIEmbeddings:
    """
    Implementa a interface necessária pelo SupabaseVectorStore:
      - embed_documents(self, texts: List[str]) -> List[List[float]]
      - embed_query(self, text: str) -> List[float]

    Usa o endpoint OpenAI-compatível da NVIDIA:
      base_url = https://integrate.api.nvidia.com/v1
      model = "baai/bge-m3"
    """
    def __init__(
        self,
        model: str = "baai/bge-m3",
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        normalize: bool = True,
        truncate: str = "NONE",
        batch_size: int = 128,
    ):
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY é obrigatório para embeddings (bge-m3).")
        self.model = model
        self.normalize = normalize
        self.truncate = truncate
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key, base_url=base_url)

   
    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for batch in batched(texts, self.batch_size):
            for attempt in range(5):
                try:
                    resp = self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                        encoding_format="float",
                    )
                    break
                except Exception as e:
                    if attempt == 4: raise
                    time.sleep(2 ** attempt)
            for item in resp.data:
                vec = item.embedding
                if self.normalize:
                    vec = _l2_normalize(vec)
                vectors.append(vec)
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Para bge-m3: NÃO prefixe documentos. Use o texto “cru”.
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # Para bge-m3: prefixe SOMENTE a consulta com instrução de query.
        q = BGE_QUERY_INSTRUCTION + text
        return self._embed([q])[0]


# =========================
# Chunking
# =========================
def chunk_documents(
    docs: List[Document],
    tokenizer_name: str,
    chunk_size_tokens: int = 380,
    chunk_overlap_tokens: int = 60,
) -> List[Document]:
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=tokenizer_name,
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
    )
    cleaned = [
        Document(page_content=clean_text(d.page_content), metadata=d.metadata)
        for d in docs
        if d.page_content and d.page_content.strip()
    ]
    chunks = splitter.split_documents(cleaned)
    for i, c in enumerate(chunks):
        # bge-m3: não adicionar "passage:" em documentos
        c.metadata = c.metadata or {}
        c.metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_idx": i,
        })
    chunks = dedup_chunks(chunks)
    return chunks


# =========================
# Ingestão
# =========================
def ingest_to_supabase(chunks: List[Document], batch_size: int = 32) -> None:
    if not SETTINGS.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY é obrigatório para embeddings NVIDIA (bge-m3).")

    print("Conectando ao Supabase...")
    client = create_client(SETTINGS.SUPABASE_URL, SETTINGS.SUPABASE_SERVICE_ROLE_KEY)

    print(f"Inicializando embeddings: {SETTINGS.EMB_MODEL} (via NVIDIA OpenAI API)")
    emb = NvidiaOpenAIEmbeddings(
        model=SETTINGS.EMB_MODEL,
        api_key=SETTINGS.NVIDIA_API_KEY,
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        normalize=True,        # recomendado para cosine
        truncate="NONE",       # evita truncar silenciosamente
        batch_size=128,
    )

    vectorstore = LCSupabaseVectorStore(
        client=client,
        table_name=SETTINGS.TABLE_NAME,
        query_name=SETTINGS.QUERY_FN,
        embedding=emb,
    )

    total = 0
    for batch in batched(chunks, batch_size):
        ids = [str(uuid.uuid4()) for _ in batch]
        vectorstore.add_documents(batch, ids=ids)
        total += len(batch)
        print(f"Inseridos {total}/{len(chunks)} chunks...")


# =========================
# Main
# =========================
def main():
    folder = os.getenv("INGEST_FOLDER", "data_pdfs")

    print("Carregando PDFs...")
    raw_docs = load_pdfs_from_folder(folder)
    print(f"Páginas carregadas: {len(raw_docs)}")

    print("Fazendo chunking por tokens (compatível com bge-m3)...")
    chunks = chunk_documents(
        raw_docs,
        tokenizer_name=SETTINGS.TOKENIZER_NAME,  # ex.: "BAAI/bge-m3"
        chunk_size_tokens=380,
        chunk_overlap_tokens=60,
    )
    print(f"Chunks gerados: {len(chunks)}")

    """ print("Validando comprimento dos chunks...") """
    """ validate_chunks(chunks, tokenizer_name=SETTINGS.TOKENIZER_NAME, max_tokens=512) """

    print("Gravando embeddings no Supabase (pgvector) em lotes...")
    ingest_to_supabase(chunks, batch_size=int(os.getenv("INGEST_BATCH_SIZE", "32")))

    print("Ingestão concluída.")


if __name__ == "__main__":
    main()