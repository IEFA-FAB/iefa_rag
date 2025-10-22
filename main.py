# main.py
from __future__ import annotations
import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag.chain import build_chain

load_dotenv()

app = FastAPI(title="RAG NVIDIA + Supabase (Híbrido)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chain, retriever = build_chain()


class AskRequest(BaseModel):
    question: str
    k: Optional[int] = None  # reservado para futuros ajustes de k em runtime


class AskResponse(BaseModel):
    answer: str
    sources: List[str]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Nota: se quiser ajustar k dinamicamente, adicione lógica aqui no retriever usado.
    answer = chain.invoke(req.question)
    docs = retriever.invoke(req.question)
    sources = []
    for d in docs:
        src = (d.metadata or {}).get("source")
        if src and src not in sources:
            sources.append(src)
    return AskResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}