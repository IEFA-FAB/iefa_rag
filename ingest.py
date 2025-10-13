# ingest_example.py
import os
from typing import List, Dict, Any
from src.ocr import parse_doc
from src.embedding import get_embedding
from src.db import upsert_document, insert_chunk

def ingest_file(path: str, lang: str = "pt", doc_type: str = "pdf", source_url: str = None):
    md, dj, chunks = parse_doc(path)

    # Opcional: embedding do documento inteiro (ou summary)
    # doc_emb = get_embedding([md[:5000]], kind="chunk")[0]
    doc_emb = None

    doc_id = upsert_document(
        filename=os.path.basename(path),
        lang=lang,
        doc_type=doc_type,
        content=md,
        source_url=source_url,
        embedding=doc_emb
    )

    texts = [c["content"] for c in chunks]
    embs = get_embedding(texts, kind="chunk")
    for i, (c, emb) in enumerate(zip(chunks, embs)):
        insert_chunk(
            document_id=doc_id,
            chunk_index=i,
            content=c["content"],
            page_num=c.get("page_num"),
            char_start=c.get("char_start") if c.get("char_start") is not None else 0,
            char_end=c.get("char_end") if c.get("char_end") is not None else 0,
            lang=c.get("lang", "pt"),
            headings=c.get("headings", []),
            is_table=bool(c.get("is_table")),
            is_code=bool(c.get("is_code")),
            embedding=emb
        )
    print(f"Ingestão concluída. document_id={doc_id}")