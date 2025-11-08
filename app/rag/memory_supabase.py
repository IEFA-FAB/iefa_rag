# app/rag/memory_supabase.py
from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from supabase import create_client, Client
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # backend only!

def _get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Configure SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def _to_lc_message(role: str, content_text: str) -> BaseMessage:
    """
    Converte papel + texto para mensagens LangChain.
    O texto deve ser a versão 'simples' (question/answer), pois é isso que entra no prompt.
    """
    if role == "user":
        return HumanMessage(content=content_text)
    if role == "assistant":
        return AIMessage(content=content_text)
    return SystemMessage(content=content_text)

def _row_to_lc_message(row: Dict[str, Any]) -> BaseMessage:
    """
    Reconstrói a mensagem LangChain a partir da linha do banco.
    Dá preferência ao content_json para extrair o texto do histórico.
    """
    role = row.get("role", "system")
    cj = row.get("content_json") or {}
    content_text = row.get("content") or ""

    # Se houver JSON estruturado, derive a string que vai no histórico:
    # - assistant: usar 'answer'
    # - user: usar 'question'
    # - system/outros: usar 'content'
    if isinstance(cj, dict):
        if role == "assistant" and isinstance(cj.get("answer"), str):
            content_text = cj.get("answer")
        elif role == "user" and isinstance(cj.get("question"), str):
            content_text = cj.get("question")
        elif isinstance(cj.get("content"), str):
            content_text = cj.get("content")

    return _to_lc_message(role, content_text)


class SupabaseChatMessageHistory(BaseChatMessageHistory):
    """
    Histórico de chat persistido no Supabase.
    Agora salva conteúdo estruturado em content_json:
      - user:      {"type":"user","question": "..."}
      - assistant: {"type":"assistant","answer":"...", "references":[...], "sources":[...]}
      - system:    {"type":"system","content":"..."}
    A coluna 'content' (text) é preenchida com a versão plana (question/answer/content)
    para compatibilidade e indexação simples.
    """

    def __init__(self, user_id: str, session_id: str, days_to_keep: int = 7, max_messages: int = 1000):
        self.user_id = user_id
        self.session_id = session_id
        self.days_to_keep = days_to_keep
        self.max_messages = max_messages
        self.client = _get_client()
        self._ensure_session()

    def _ensure_session(self):
        # Garante que a sessão exista e pertença a este user_id
        resp = (
            self.client.table("chat_sessions")
            .select("id, user_id")
            .eq("id", self.session_id)
            .limit(1)
            .execute()
        )
        row = (resp.data or [None])[0]
        if row:
            if row["user_id"] != self.user_id:
                # Defesa em profundidade: se a sessão existir, mas não for do usuário, bloqueia
                raise PermissionError("Sessão não pertence a este usuário")
            return

        self.client.table("chat_sessions").insert(
            {"id": self.session_id, "user_id": self.user_id}
        ).execute()

    def _now_utc_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _default_expiry(self) -> str:
        return (datetime.now(timezone.utc) + timedelta(days=self.days_to_keep)).isoformat()

    def _touch_session(self) -> None:
        # Atualiza last_message_at
        self.client.table("chat_sessions").update(
            {"last_message_at": self._now_utc_iso()}
        ).eq("id", self.session_id).eq("user_id", self.user_id).execute()

    # ------------ Novos métodos estruturados ------------

    def add_user_question(self, question: str) -> None:
        payload = {
            "type": "user",
            "question": question or "",
        }
        self.client.table("chat_messages").insert(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "role": "user",
                "content": question or "",
                "content_json": payload,
                "expires_at": self._default_expiry(),
            }
        ).execute()
        self._touch_session()

    def add_assistant_answer(
        self,
        answer: str,
        references: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None,
        token_count: Optional[int] = None,
    ) -> None:
        payload = {
            "type": "assistant",
            "answer": answer or "",
            "references": references or [],
            "sources": sources or [],
        }
        row = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "role": "assistant",
            "content": answer or "",
            "content_json": payload,
            "expires_at": self._default_expiry(),
        }
        if token_count is not None:
            row["token_count"] = int(token_count)
        self.client.table("chat_messages").insert(row).execute()
        self._touch_session()

    def add_system_message(self, content: str) -> None:
        payload = {
            "type": "system",
            "content": content or "",
        }
        self.client.table("chat_messages").insert(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "role": "system",
                "content": content or "",
                "content_json": payload,
                "expires_at": self._default_expiry(),
            }
        ).execute()
        self._touch_session()

    # ------------ Compatibilidade (API anterior) ------------

    def add_message(self, message: BaseMessage) -> None:
        """
        Mantida por compatibilidade. Se possível, use add_user_question/add_assistant_answer/add_system_message.
        """
        role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else "system"
        content_text = message.content if isinstance(message.content, str) else str(message.content)

        if role == "user":
            self.add_user_question(content_text)
        elif role == "assistant":
            # Sem referências/sources disponíveis por esta via
            self.add_assistant_answer(content_text, references=[], sources=[])
        else:
            self.add_system_message(content_text)

    def clear(self) -> None:
        self.client.table("chat_messages").delete().eq("session_id", self.session_id).eq("user_id", self.user_id).execute()

    @property
    def messages(self) -> List[BaseMessage]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.days_to_keep)
        resp = (
            self.client.table("chat_messages")
            .select("role, content, content_json, created_at")
            .eq("user_id", self.user_id)
            .eq("session_id", self.session_id)
            .gte("created_at", cutoff.isoformat())
            .order("created_at", desc=False)
            .limit(self.max_messages)
            .execute()
        )
        rows = resp.data or []
        return [_row_to_lc_message(r) for r in rows]