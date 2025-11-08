# main.py
from __future__ import annotations
import os
import json
import uuid
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.rag.chain import build_chain
from app.rag.memory_supabase import _get_client

load_dotenv()

app = FastAPI(title="RAG NVIDIA + Supabase (Híbrido)")

ALLOWED_ORIGINS = [
    "https://portal.iefa.com.br",
    # ambientes de dev (se necessário):
    "http://localhost:3000",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,          # não use "*"
    allow_credentials=True,                 # necessário se usa cookies/credenciais
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
    allow_headers=[
        "Content-Type", "Authorization", "Accept", "X-User-Id", "X-Requested-With"
    ],
)

ANON_COOKIE = os.getenv("ANON_COOKIE", "anon_id")

# Namespaces fixos para derivar UUID v5 (podem ser ajustados por env)
DEFAULT_ANON_NS = uuid.UUID(os.getenv("ANON_NAMESPACE_UUID", "5f6a1cf1-2e58-4c49-9c4d-3c3a4a9b7a6f"))
DEFAULT_AUTH_NS = uuid.UUID(os.getenv("AUTH_NAMESPACE_UUID", "76a2a4a3-1e22-4d2b-8f0e-9a6b6010d7a1"))


class AskRequest(BaseModel):
    question: str
    user_id: Optional[str] = None       # IGNORADO para persistência (mantido por compatibilidade)
    session_id: Optional[str] = None    # cliente pode enviar para continuar a sessão
    k: Optional[int] = None             # reservado p/ uso futuro


class Reference(BaseModel):
    n: int
    source: str
    page: Optional[int] = None
    snippet: Optional[str] = None
    rank: Optional[float] = None
    doc_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    references: List[Reference]
    sources: List[str]
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


def _cookie_secure(request: Request) -> bool:
    try:
        return request.url.scheme == "https"
    except Exception:
        return True


def get_identity(request: Request, response: Response) -> Tuple[str, bool]:
    """
    Retorna (identity, is_authenticated).
    identity:
      - autenticado: idealmente seria o auth.uid() (UUID). Aqui aceitamos X-User-Id de um gateway confiável.
      - anônimo: 'anon:<uuid-v4>' persistido em cookie httpOnly.
    """
    x_uid = request.headers.get("X-User-Id")
    if x_uid:
        # Considera autenticado se um gateway confiável injetou X-User-Id (após validar o JWT).
        return x_uid, True

    anon_id = request.cookies.get(ANON_COOKIE)
    if not anon_id:
        anon_id = f"anon:{uuid.uuid4()}"
        response.set_cookie(
            key=ANON_COOKIE,
            value=anon_id,
            httponly=True,
            samesite="lax",
            secure=_cookie_secure(request),
            path="/",
            max_age=60 * 60 * 24 * 30,  # 30 dias
        )
    return anon_id, False


def to_db_user_id(identity: str, is_auth: bool) -> str:
    """
    Converte qualquer identidade para um UUID string para persistir no banco (coluna user_id UUID).
    - Se identity já é UUID, retorna como está.
    - Caso contrário, deriva um UUID v5 determinístico a partir de um namespace:
        * anônimo -> DEFAULT_ANON_NS
        * autenticado não-UUID (ex.: ambiente de dev com X-User-Id arbitrary) -> DEFAULT_AUTH_NS
    """
    try:
        return str(uuid.UUID(identity))
    except Exception:
        ns = DEFAULT_AUTH_NS if is_auth else DEFAULT_ANON_NS
        return str(uuid.uuid5(ns, identity))


def get_or_create_single_anon_session(client, db_user_id: str, preferred_session_id: Optional[str]) -> str:
    """
    Para anônimos: garante que exista somente 1 sessão por 'db_user_id'.
    """
    existing = (
        client.table("chat_sessions")
        .select("id")
        .eq("user_id", db_user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    if existing:
        return existing[0]["id"]

    sid = preferred_session_id or str(uuid.uuid4())
    client.table("chat_sessions").insert({"id": sid, "user_id": db_user_id}).execute()
    return sid


def ensure_session_for_user(client, db_user_id: str, session_id: str) -> str:
    """
    Para usuários autenticados (ou ids normalizados):
    - Se a sessão existe e pertence ao db_user_id, ok.
    - Se não existe, cria para o db_user_id.
    - Se pertence a outro user, 403.
    """
    resp = (
        client.table("chat_sessions")
        .select("id, user_id")
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    row = (resp.data or [None])[0]
    if row:
        if row["user_id"] != db_user_id:
            raise HTTPException(status_code=403, detail="Sessão não pertence ao usuário")
        return session_id

    client.table("chat_sessions").insert({"id": session_id, "user_id": db_user_id}).execute()
    return session_id


def require_session_ownership(client, db_user_id: str, session_id: str) -> None:
    """
    Garante que a sessão exista e pertença ao db_user_id.
    """
    resp = (
        client.table("chat_sessions")
        .select("id, user_id")
        .eq("id", session_id)
        .limit(1)
        .execute()
    )
    row = (resp.data or [None])[0]
    if not row:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    if row["user_id"] != db_user_id:
        raise HTTPException(status_code=403, detail="Sessão não pertence a este usuário")


@app.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
def get_session_messages(session_id: str, request: Request, response: Response, user_id: Optional[str] = None):
    """
    Restrito a usuários autenticados.
    Anônimos não podem buscar histórico.
    """
    identity, is_auth = get_identity(request, response)
    if not is_auth:
        raise HTTPException(status_code=403, detail="Anônimos não podem buscar histórico")

    db_user_id = to_db_user_id(identity, is_auth=True)
    client = _get_client()
    require_session_ownership(client, db_user_id, session_id)

    rows = (
        client.table("chat_messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .eq("user_id", db_user_id)
        .order("created_at", desc=False)
        .limit(1000)
        .execute()
        .data
        or []
    )
    return rows


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request, response: Response, user_id: Optional[str] = None):
    """
    Pode ser usado por autenticados e anônimos, mas somente para deletar a própria sessão.
    """
    identity, is_auth = get_identity(request, response)
    db_user_id = to_db_user_id(identity, is_auth)
    client = _get_client()

    require_session_ownership(client, db_user_id, session_id)

    client.table("chat_messages").delete().eq("session_id", session_id).eq("user_id", db_user_id).execute()
    client.table("chat_sessions").delete().eq("id", session_id).eq("user_id", db_user_id).execute()
    return {"ok": True}


@app.get("/sessions", response_model=List[SessionOut])
def list_sessions(request: Request, response: Response, user_id: Optional[str] = None):
    """
    Restrito a usuários autenticados.
    Anônimos não podem listar sessões.
    """
    identity, is_auth = get_identity(request, response)
    if not is_auth:
        raise HTTPException(status_code=403, detail="Anônimos não podem listar sessões")

    db_user_id = to_db_user_id(identity, is_auth=True)
    client = _get_client()
    rows = (
        client.table("chat_sessions")
        .select("id, user_id, created_at, last_message_at")
        .eq("user_id", db_user_id)
        .order("last_message_at", desc=True)
        .limit(50)
        .execute()
        .data
        or []
    )
    return rows


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request, response: Response):
    """
    Resposta não-stream.
    - Anônimo: 1 sessão única por anon_id (cookie httpOnly).
    - Autenticado: valida/gera a sessão garantindo propriedade.
    Todos os acessos ao banco usam um user_id UUID (normalizado).
    """
    identity, is_auth = get_identity(request, response)
    db_user_id = to_db_user_id(identity, is_auth)
    client = _get_client()

    # Resolve session_id conforme política
    if is_auth:
        session_id = req.session_id or str(uuid.uuid4())
        session_id = ensure_session_for_user(client, db_user_id, session_id)
    else:
        session_id = get_or_create_single_anon_session(client, db_user_id, req.session_id)

    # Constrói a chain com user_id normalizado (UUID) e session_id
    chain_dict = build_chain(user_id=db_user_id, session_id=session_id)

    result = chain_dict["invoke"](req.question)
    references = [Reference(**r) for r in result.get("references", [])]
    return AskResponse(
        answer=result.get("answer", ""),
        references=references,
        sources=result.get("sources", []),
        session_id=session_id,
    )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, request: Request, response: Response):
    """
    Streaming SSE (text/event-stream).
    """
    identity, is_auth = get_identity(request, response)
    db_user_id = to_db_user_id(identity, is_auth)
    client = _get_client()

    if is_auth:
        session_id = req.session_id or str(uuid.uuid4())
        session_id = ensure_session_for_user(client, db_user_id, session_id)
    else:
        session_id = get_or_create_single_anon_session(client, db_user_id, req.session_id)

    chain_dict = build_chain(user_id=db_user_id, session_id=session_id)

    async def event_generator():
        try:
            async for chunk in chain_dict["astream"](req.question):
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'type': 'token', 'delta': chunk})}\n\n"
                elif isinstance(chunk, dict) and "answer" in chunk:
                    payload = {"type": "final", **chunk, "session_id": session_id}
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "event: end\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}