# query_service.py
from typing import List, Dict, Any, Optional
from search import hybrid_search
from rerank import rerank
from generator import generate_answer

def select_mode(query: str) -> str:
    q = query.lower()
    if len(q) > 140 or any(k in q for k in ["overview","panorama","relações","mapa conceitual","como se conectam"]):
        return "graph_first"
    if any(k in q for k in ["definição","onde","qual","cite","página","offset"]):
        return "hybrid_first"
    return "auto"

def answer_query(
    question: str,
    lang_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    match_count: int = 20,
    use_graph: bool = False
) -> Dict[str, Any]:
    mode = select_mode(question)

    # 1) Híbrido (sempre rodamos, e mesclamos depois se for graph_first)
    candidates = hybrid_search(
        question,
        match_count=match_count,
        full_text_weight=1.2,
        semantic_weight=1.0,
        rrf_k=50,
        lang_filter=lang_filter,
        doc_type_filter=doc_type_filter
    )

    # 2) Rerank opcional
    ranked = rerank(question, candidates, top_n=8)

    # 3) GraphRAG opcional (stub)
    graph_context: List[Dict[str, Any]] = []
    if use_graph or mode == "graph_first":
        # TODO: chamar GraphRAG (global/local/DRIFT) e produzir passagens/summary
        graph_context = []  # Ex.: [{"chunk_id": -1, "page": None, "char_start": 0, "char_end": 0, "content": "Resumo da comunidade ..."}]

    # 4) Preparar contexto para geração (anexar chunk_id/página/offsets)
    context_items: List[Dict[str, Any]] = []
    for c in ranked:
        context_items.append({
            "chunk_id": c["id"],
            "page": c.get("page_num"),
            "char_start": c.get("char_start"),
            "char_end": c.get("char_end"),
            "content": c["content"],
        })
    context_items.extend(graph_context)

    # 5) Geração com Gemini
    result = generate_answer(context_items, question)
    return result