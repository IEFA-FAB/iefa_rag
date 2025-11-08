# main.py (modificado)
from __future__ import annotations
import os
import json
import uuid
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag.chain import build_chain
from app.rag.memory_supabase import _get_client  # MODIFICADO: agora importamos build_chain diretamente

load_dotenv()

app = FastAPI(title="RAG NVIDIA + Supabase (Híbrido)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODIFICADO: Removemos a inicialização global da chain


class AskRequest(BaseModel):
    question: str
    user_id: Optional[str] = None      # fornecido pelo cliente ou derivado do JWT
    session_id: Optional[str] = None   # fornecido pelo cliente para continuar sessão
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
    # ADICIONADO: session_id para que o cliente possa continuar a conversa
    session_id: str



class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str

class SessionOut(BaseModel):
    id: str
    user_id: str
    created_at: str
    last_message_at: Optional[str] = None

@app.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
def get_session_messages(session_id: str, user_id: Optional[str] = None):
    # Ideal: derivar o user_id de um JWT (Supabase Auth). Aqui deixo como query param para simplicidade.
    uid = user_id or "anonymous"
    client = _get_client()

    # garante que a sessão pertence ao user_id
    sess = client.table("chat_sessions").select("id").eq("id", session_id).eq("user_id", uid).limit(1).execute()
    if not sess.data:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    rows = (
        client.table("chat_messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .eq("user_id", uid)
        .order("created_at", desc=False)
        .limit(1000)
        .execute()
        .data or []
    )
    return rows

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, user_id: Optional[str] = None):
    uid = user_id or "anonymous"
    client = _get_client()

    # apagar mensagens + sessão (ON DELETE CASCADE recomendado no schema)
    client.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", uid).execute()
    client.table("chat_sessions").delete().eq("id", session_id).eq("user_id", uid).execute()
    return {"ok": True}

@app.get("/sessions", response_model=List[SessionOut])
def list_sessions(user_id: Optional[str] = None):
    uid = user_id or "anonymous"
    client = _get_client()
    rows = (
        client.table("chat_sessions")
        .select("id, user_id, created_at, last_message_at")
        .eq("user_id", uid)
        .order("last_message_at", desc=True)
        .limit(50)
        .execute()
        .data or []
    )
    return rows

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Resposta não-stream. Agora a chain retorna um dicionário com:
      - answer (string com [n] e seção 'Referências' já enxuta)
      - references (lista com n/source/page/snippet/...)
      - sources (apenas as fontes efetivamente citadas)
    """
    # MODIFICADO: Obter user_id e session_id da requisição
    user_id = req.user_id or "anonymous"  # ideal: extrair do JWT do Supabase
    session_id = req.session_id or str(uuid.uuid4())
    
    # MODIFICADO: Construir a chain com user_id e session_id
    chain_dict = build_chain(user_id=user_id, session_id=session_id)
    
    # MODIFICADO: Usar o wrapper invoke_with_memory
    result = chain_dict["invoke"](req.question)
    
    # result esperado: {"answer": str, "references": List[dict], "sources": List[str]}
    references = [Reference(**r) for r in result.get("references", [])]
    return AskResponse(
        answer=result.get("answer", ""),
        references=references,
        sources=result.get("sources", []),
        session_id=session_id,
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
    # MODIFICADO: Obter user_id e session_id da requisição
    user_id = req.user_id or "anonymous"
    session_id = req.session_id or str(uuid.uuid4())
    
    # MODIFICADO: Construir a chain com user_id e session_id
    chain_dict = build_chain(user_id=user_id, session_id=session_id)

    async def event_generator():
        full_answer = ""
        try:
            # MODIFICADO: Usar o wrapper astream_with_memory
            async for chunk in chain_dict["astream"](req.question):
                if isinstance(chunk, str):
                    full_answer += chunk
                    yield f"data: {json.dumps({'type':'token','delta':chunk})}\n\n"
                elif isinstance(chunk, dict) and "answer" in chunk:
                    full_answer = chunk["answer"]
                    yield f"data: {json.dumps({'type':'final', **chunk, 'session_id': session_id})}\n\n"
                    # REMOVER estas duas linhas:
                    # chain_dict["memory"].save_context({"question": req.question}, {"answer": full_answer})
                    yield "event: end\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            yield "event: end\ndata: {}\n\n"

    return StreamingResponse(event_generator(),media_type="text/event-stream",headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no", },)


@app.get("/health")
def health():
    return {"status": "ok"}