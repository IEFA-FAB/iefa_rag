# graph.py
from typing import List, Dict, Any

def extract_entities_relations(text: str):
    # TODO: Implementar com LLM+NLP
    # Retorne listas do tipo:
    # entities = [{"label": "X", "type": "ORG", "properties": {...}}]
    # relations = [{"source": "X", "target": "Y", "relation": "rel", "weight": 1.0}]
    return [], []

def upsert_nodes(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # TODO: inserir em graph_nodes (tabela); retornar registros com id
    return []

def upsert_edges(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # TODO: inserir em graph_edges; retornar registros com id
    return []

def detect_communities(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # TODO: implementar detecção (Leiden/Louvain). Stub: 1 comunidade única
    return [{"community_id": "C0", "nodes": nodes, "edges": edges}]

def summarize_communities(communities: List[Dict[str, Any]], max_tokens=400) -> List[Dict[str, Any]]:
    # TODO: usar LLM para sumarizar cada comunidade
    return [{"community_id": c["community_id"], "summary": "Resumo stub", "tokens": 50, "metadata": {}} for c in communities]

def upsert_community_summaries(summaries: List[Dict[str, Any]]):
    # TODO: inserir em community_summaries
    return

def build_graph_from_chunks(chunk_rows: List[Dict[str, Any]]):
    entities, relations = [], []
    for ch in chunk_rows:
        es, rs = extract_entities_relations(ch["content"])
        entities.extend(es); relations.extend(rs)
    nodes = upsert_nodes(entities)
    edges = upsert_edges(relations)
    communities = detect_communities(nodes, edges)
    summaries = summarize_communities(communities, max_tokens=400)
    upsert_community_summaries(summaries)