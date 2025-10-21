import os
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
TABLE_NAME = os.getenv("TABLE_NAME", "documents")
QUERY_FN = os.getenv("QUERY_FN", "match_documents")
EMB_MODEL = os.getenv("EMB_MODEL", "nvidia/nv-embedqa-e5-v5")

def get_semantic_retriever(k=5):
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    emb = NVIDIAEmbeddings(model=EMB_MODEL)
    vectorstore = SupabaseVectorStore(
        client=supabase, embedding=emb,
        table_name=TABLE_NAME, query_name=QUERY_FN
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

def get_bm25_retriever(k=5):
    # Para BM25, buscamos os textos no Supabase e montamos o índice em memória
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    rows = supabase.table(TABLE_NAME).select("content,metadata").execute().data
    docs = [Document(page_content=r["content"] or "", metadata=r.get("metadata") or {}) for r in rows if r.get("content")]
    retr = BM25Retriever.from_documents(docs)
    retr.k = k
    return retr

def get_hybrid_retriever(k_sem=4, k_bm25=6, weights=(0.5, 0.5)):
    sem = get_semantic_retriever(k=k_sem)
    bm25 = get_bm25_retriever(k=k_bm25)
    hybrid = EnsembleRetriever(
        retrievers=[bm25, sem],
        weights=list(weights)  # Reciprocal Rank Fusion (RRF) sob o capô
    )
    return hybrid

if __name__ == "__main__":
    hybrid = get_hybrid_retriever()
    q = "Qual é o procedimento descrito no documento X?"
    docs = hybrid.invoke(q)
    print("Top documentos recuperados (híbrido):")
    for i, d in enumerate(docs, 1):
        print(f"{i:02d} - {d.metadata.get('source')}: {d.page_content[:120]}...")