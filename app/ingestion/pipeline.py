# app/ingestion/pipeline.py
from __future__ import annotations
import os
import re
import glob
import uuid
from typing import List, Iterable

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore as LCSupabaseVectorStore
from langchain.schema import Document
from transformers import AutoTokenizer
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from app.config import SETTINGS


def batched(iterable: List, n: int) -> Iterable[List]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n', '', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def apply_e5_prefix(text: str, is_document: bool = True) -> str:
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


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def chunk_documents(
    docs: List[Document],
    tokenizer_name: str = SETTINGS.TOKENIZER_NAME,
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
    tokenizer_name: str = SETTINGS.TOKENIZER_NAME,
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
        print("Exemplos:", too_long[:show_examples])


def ingest_to_supabase(
    chunks: List[Document],
    batch_size: int = 100,
) -> None:
    if not SETTINGS.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY é obrigatório para embeddings NVIDIA.")

    print("Conectando ao Supabase...")
    client = create_client(SETTINGS.SUPABASE_URL, SETTINGS.SUPABASE_SERVICE_ROLE_KEY)

    print(f"Inicializando embeddings NVIDIA: {SETTINGS.EMB_MODEL}")
    emb = NVIDIAEmbeddings(model=SETTINGS.EMB_MODEL, api_key=SETTINGS.NVIDIA_API_KEY, truncate="END")

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


def main():
    folder = os.getenv("INGEST_FOLDER", "data_pdfs")
    print("Carregando PDFs...")
    raw_docs = load_pdfs_from_folder(folder)
    print(f"Páginas carregadas: {len(raw_docs)}")

    print("Fazendo chunking por tokens (compatível com E5)...")
    chunks = chunk_documents(
        raw_docs,
        tokenizer_name=SETTINGS.TOKENIZER_NAME,
        chunk_size_tokens=380,
        chunk_overlap_tokens=60
    )
    print(f"Chunks gerados: {len(chunks)}")

    print("Validando comprimento dos chunks...")
    validate_chunks(chunks, tokenizer_name=SETTINGS.TOKENIZER_NAME, max_tokens=512)

    print("Gravando embeddings no Supabase (pgvector) em lotes...")
    ingest_to_supabase(chunks, batch_size=int(os.getenv("INGEST_BATCH_SIZE", "100")))

    print("Ingestão concluída.")


if __name__ == "__main__":
    main()