# search.py
import os
import psycopg2
from typing import List, Dict, Any, Optional
from embedding import get_embedding

DATABASE_URL = os.environ["DATABASE_URL"]

def _vector_literal(vec: List[float]) -> str:
    # pgvector aceita literal tipo '[0.1, 0.2, ...]'
    return "[" + ", ".join(f"{x:.7f}" for x in vec) + "]"

def hybrid_search(
    query_text: str,
    match_count: int = 20,
    full_text_weight: float = 1.2,
    semantic_weight: float = 1.0,
    rrf_k: int = 50,
    lang_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    semantic_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    qemb = get_embedding([query_text], kind="query")[0]  # 1024-dim normalizado
    qlit = _vector_literal(qemb)

    sql = """
    select id, document_id, content, page_num, char_start, char_end, lang
    from hybrid_search_chunks(
      query_text := %s,
      query_embedding := %s::vector,
      match_count := %s,
      full_text_weight := %s,
      semantic_weight := %s,
      rrf_k := %s,
      lang_filter := %s,
      doc_type_filter := %s
    )
    """
    rows = []
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                query_text, qlit, match_count, full_text_weight,
                semantic_weight, rrf_k, lang_filter, doc_type_filter
            ))
            for r in cur.fetchall():
                rows.append({
                    "id": r[0],
                    "document_id": r[1],
                    "content": r[2],
                    "page_num": r[3],
                    "char_start": r[4],
                    "char_end": r[5],
                    "lang": r[6],
                })
    # Nota: a função não retorna score — se quiser threshold semântico,
    # você pode recalc similiaridade com qemb e o embedding do chunk (exigir join/consulta extra)
    # ou incluir score na função SQL. Aqui apenas retornamos rows direto.
    return rows