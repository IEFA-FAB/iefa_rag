import os, glob, uuid
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
TABLE_NAME = os.getenv("TABLE_NAME", "documents")
QUERY_FN = os.getenv("QUERY_FN", "match_documents")
EMB_MODEL = os.getenv("EMB_MODEL", "nvidia/nv-embedqa-e5-v5")

def load_pdfs_from_folder(folder: str):
    docs = []
    pdf_paths = glob.glob(os.path.join(folder, "*.pdf"))
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for d in pages:
            d.metadata = d.metadata or {}
            d.metadata.update({"source": os.path.basename(path)})
            docs.append(d)
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def main():
    folder = "data_pdfs"   # coloque seus PDFs nesta pasta
    print("Carregando PDFs...")
    raw_docs = load_pdfs_from_folder(folder)
    print(f"PDFs carregados; páginas: {len(raw_docs)}")

    print("Fazendo chunking...")
    chunks = chunk_documents(raw_docs)
    print(f"Chunks gerados: {len(chunks)}")

    print("Conectando no Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    emb = NVIDIAEmbeddings(model=EMB_MODEL)

    print("Gravando embeddings no Supabase (pgvector)...")
    _ = SupabaseVectorStore.from_documents(
        documents=chunks,
        embedding=emb,
        client=supabase,
        table_name=TABLE_NAME,
        query_name=QUERY_FN,
    )
    print("Ingestão concluída.")

if __name__ == "__main__":
    main()