# main.py
from __future__ import annotations
import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


class Reference(BaseModel):
    n: int
    source: str
    page: Optional[int] = None
    snippet: Optional[str] = None
    rank: Optional[float] = None
    doc_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    # referências efetivamente usadas no texto (deduplicadas por [n] e na ordem de ocorrência)
    references: List[Reference]
    # nomes únicos das fontes realmente citadas no answer (deduplicados)
    sources: List[str]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Resposta não-stream. Agora a chain retorna um dicionário com:
      - answer (string com [n] e seção 'Referências' já enxuta)
      - references (lista com n/source/page/snippet/...)
      - sources (apenas as fontes efetivamente citadas)
    """
    result = chain.invoke(req.question)
    # result esperado: {"answer": str, "references": List[dict], "sources": List[str]}
    references = [Reference(**r) for r in result.get("references", [])]
    return AskResponse(
        answer=result.get("answer", ""),
        references=references,
        sources=result.get("sources", []),
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    Streaming SSE (text/event-stream).
    Eventos:
      - {"type": "token", "delta": "..."} : tokens chegando em tempo real
      - {"type": "final", "answer": str, "references": [...], "sources": [...]}
      - event: end (encerra o stream)
      - event: error (em caso de erro)

    Observação: garanta que seu frontend aceite SSE e não bufferize a resposta.
    """
    async def event_generator():
        try:
            # Usamos astream em vez de astream_events para obter o stream de tokens diretamente
            async for chunk in chain.astream(req.question):
                # Verifica se é um chunk de token
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'type': 'token', 'delta': chunk})}\n\n"
                # Verifica se é o resultado final
                elif isinstance(chunk, dict) and "answer" in chunk:
                    yield f"data: {json.dumps({'type': 'final', **chunk})}\n\n"
                    yield "event: end\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            yield "event: end\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}