import os
import glob
import uuid
from typing import List, Iterable

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client
from langchain_community.document_loaders import PyPDFLoader  # ou PyMuPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.schema import Document
from transformers import AutoTokenizer

import re

# ----------------------------
# Config e validações
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
TABLE_NAME = os.getenv("TABLE_NAME", "documents")
QUERY_FN = os.getenv("QUERY_FN", "match_documents")
EMB_MODEL = os.getenv("EMB_MODEL", "nvidia/nv-embedqa-e5-v5")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "intfloat/e5-large-v2")  # alinhado ao E5
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")  # necessário para NVIDIAEmbeddings

REQUIRED_VARS = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    "NVIDIA_API_KEY": NVIDIA_API_KEY,  # se usar NVIDIA
}
missing = [k for k, v in REQUIRED_VARS.items() if not v]
if missing:
    raise RuntimeError(f"Variáveis de ambiente faltando: {missing}")

# Importa aqui para poder passar api_key, se necessário
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# ----------------------------
# Utilitários
# ----------------------------
def batched(iterable: List, n: int) -> Iterable[List]:
    """Gera lotes (batches) de tamanho n."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def clean_text(text: str) -> str:
    """Normalizações simples para melhorar a qualidade da indexação."""
    text = re.sub(r'-\s*\n', '', text)     # junta palavras hifenizadas no fim da linha
    text = re.sub(r'\s*\n\s*', '\n', text) # normaliza quebras de linha
    text = re.sub(r'\s+', ' ', text)       # colapsa espaços
    return text.strip()

def apply_e5_prefix(text: str, is_document: bool = True) -> str:
    """Aplica prefixo E5 quando pertinente."""
    prefix = "passage: " if is_document else "query: "
    return prefix + text

def dedup_chunks(chunks: List[Document]) -> List[Document]:
    seen = set()
    unique = []
    for c in chunks:
        key = hash(c.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

# ----------------------------
# Carregamento de PDFs
# ----------------------------
def load_pdfs_from_folder(folder: str) -> List[Document]:
    docs = []
    pdf_paths = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)  # ou PyMuPDFLoader
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

# ----------------------------
# Chunking/tokenização
# ----------------------------
def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def chunk_documents(
    docs: List[Document],
    tokenizer_name: str = TOKENIZER_NAME,
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
    # Prefixo E5 para documentos + metadados úteis
    for i, c in enumerate(chunks):
        c.page_content = apply_e5_prefix(c.page_content, is_document=True)
        c.metadata = c.metadata or {}
        c.metadata.update({
            "chunk_id": str(uuid.uuid4()),
            "chunk_idx": i,
        })
        
    chunks = dedup_chunks(chunks)
    return chunks

def validate_chunks(
    chunks: List[Document],
    tokenizer_name: str = TOKENIZER_NAME,
    max_tokens: int = 512,
    show_examples: int = 5,
) -> None:
    tok = get_tokenizer(tokenizer_name)
    too_long = []
    for i, c in enumerate(chunks):
        # não adicione special tokens na contagem
        n = len(tok(c.page_content, add_special_tokens=False)["input_ids"])
        if n > max_tokens:
            too_long.append((i, n))
    print(f"Chunks acima de {max_tokens} tokens: {len(too_long)}")
    if too_long[:show_examples]:
        print("Exemplos:", too_long[:show_examples])

# ----------------------------
# Ingestão Supabase
# ----------------------------
def ingest_to_supabase(
    chunks: List[Document],
    table_name: str = TABLE_NAME,
    query_fn: str = QUERY_FN,
    emb_model: str = EMB_MODEL,
    batch_size: int = 100,
) -> None:
    print("Conectando ao Supabase...")
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    print(f"Inicializando embeddings NVIDIA: {emb_model}")
    emb = NVIDIAEmbeddings(model=emb_model, api_key=NVIDIA_API_KEY, truncate="END")

    # Use a instância do vectorstore para adicionar em lotes
    vectorstore = SupabaseVectorStore(
        client=client,
        table_name=table_name,
        query_name=query_fn,
        embedding=emb,
    )

    total = 0
    for batch in batched(chunks, batch_size):
        ids = [str(uuid.uuid4()) for _ in batch]
        vectorstore.add_documents(batch, ids=ids)
        total += len(batch)
        print(f"Inseridos {total}/{len(chunks)} chunks...")

# ----------------------------
# Main
# ----------------------------
def main():
    folder = "data_pdfs"
    print("Carregando PDFs...")
    raw_docs = load_pdfs_from_folder(folder)
    print(f"Páginas carregadas: {len(raw_docs)}")

    print("Fazendo chunking por tokens (compatível com E5)...")
    chunks = chunk_documents(
        raw_docs,
        tokenizer_name=TOKENIZER_NAME,
        chunk_size_tokens=380,
        chunk_overlap_tokens=60
    )
    print(f"Chunks gerados: {len(chunks)}")

    print("Validando comprimento dos chunks...")
    validate_chunks(chunks, tokenizer_name=TOKENIZER_NAME, max_tokens=512)

    print("Gravando embeddings no Supabase (pgvector) em lotes...")
    ingest_to_supabase(chunks, batch_size=100)

    print("Ingestão concluída.")

if __name__ == "__main__":
    main()