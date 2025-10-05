# embedding.py
import os
import json
import hashlib
import numpy as np
from typing import List, Literal, Optional
from redis import Redis

# ENV esperadas:
# REDIS_URL
# EMBEDDING_PROVIDER = (bge|openai|google|voyage)
# EMBEDDING_MODEL = ex.: bge-m3 | text-embedding-3-large | text-embedding-004 | voyage-multilingual-2

redis = Redis.from_url(os.environ["REDIS_URL"])
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
DEFAULT_PROVIDER: Literal["bge","openai","google","voyage"] = os.getenv("EMBEDDING_PROVIDER", "bge")  # bge local/hosted

def cache_key(kind: Literal["chunk","query"], text: str, model: str = DEFAULT_MODEL) -> str:
    h = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    return f"emb:{kind}:{model}:{h}"

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

def embed_provider(texts: List[str],
                   provider: str = DEFAULT_PROVIDER,
                   model: str = DEFAULT_MODEL) -> List[List[float]]:
    """
    Troque a implementação abaixo pelo seu provedor real.
    - bge: chame o endpoint local/hosted de bge-m3
    - openai: use client.embeddings.create(model=model, input=texts)
    - google: embeddings via Vertex/AI Studio (text-embedding-004)
    - voyage: voyageai.Embeddings
    """
    if provider == "openai":
        # from openai import OpenAI
        # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # resp = client.embeddings.create(model=model, input=texts)
        # return [d.embedding for d in resp.data]
        raise NotImplementedError("TODO: implemente OpenAI embeddings")
    elif provider == "google":
        # Google text-embedding-004
        raise NotImplementedError("TODO: implemente Google embeddings")
    elif provider == "voyage":
        # import voyageai
        # vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        # resp = vo.embed(texts, model=model)
        # return resp.embeddings
        raise NotImplementedError("TODO: implemente Voyage embeddings")
    else:
        # bge-m3 local/hosted — placeholder
        # Exemplo: chamada HTTP a um servidor local
        # return http_embed(texts, endpoint=os.environ["BGE_ENDPOINT"])
        raise NotImplementedError("TODO: implemente BGE embeddings")

def get_embedding(texts: List[str],
                  kind: Literal["chunk","query"] = "chunk",
                  provider: str = DEFAULT_PROVIDER,
                  model: str = DEFAULT_MODEL,
                  normalize_vectors: bool = True) -> List[List[float]]:
    out: List[Optional[List[float]]] = []
    to_compute: List[str] = []
    idx_map = {}

    for i, t in enumerate(texts):
        key = cache_key(kind, t, model)
        blob = redis.get(key)
        if blob:
            out.append(json.loads(blob))
        else:
            out.append(None)
            idx_map[i] = t
            to_compute.append(t)

    if to_compute:
        embeds = embed_provider(to_compute, provider=provider, model=model)
        for (i, t), emb in zip(idx_map.items(), embeds):
            v = np.array(emb, dtype=np.float32)
            if normalize_vectors:
                v = normalize(v)
            lst = v.tolist()
            out[i] = lst
            redis.setex(cache_key(kind, t, model), 2592000, json.dumps(lst))

    # type: ignore
    return out  # List[List[float]]