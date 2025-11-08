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

def _to_lc_message(role: str, content: str) -> BaseMessage:
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return SystemMessage(content=content)

class SupabaseChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, user_id: str, session_id: str, days_to_keep: int = 7, max_messages: int = 1000):
        self.user_id = user_id
        self.session_id = session_id
        self.days_to_keep = days_to_keep
        self.max_messages = max_messages
        self.client = _get_client()
        self._ensure_session()

    def _ensure_session(self):
        # garanta que a sessão exista
        resp = (
            self.client.table("chat_sessions")
            .select("id")
            .eq("id", self.session_id)
            .eq("user_id", self.user_id)
            .limit(1)
            .execute()
        )
        if not resp.data:
            self.client.table("chat_sessions").insert(
                {"id": self.session_id, "user_id": self.user_id}
            ).execute()

    def add_message(self, message: BaseMessage) -> None:
        role = "user" if isinstance(message, HumanMessage) else "assistant" if isinstance(message, AIMessage) else "system"
        content = message.content if isinstance(message.content, str) else str(message.content)
        expires_at = datetime.now(timezone.utc) + timedelta(days=self.days_to_keep)

        self.client.table("chat_messages").insert(
            {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "role": role,
                "content": content,
                "expires_at": expires_at.isoformat(),
            }
        ).execute()

        # atualiza last_message_at na sessão
        self.client.table("chat_sessions").update(
            {"last_message_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", self.session_id).eq("user_id", self.user_id).execute()

    def clear(self) -> None:
        self.client.table("chat_messages").delete().eq("session_id", self.session_id).eq("user_id", self.user_id).execute()

    @property
    def messages(self) -> List[BaseMessage]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.days_to_keep)
        resp = (
            self.client.table("chat_messages")
            .select("role, content, created_at")
            .eq("user_id", self.user_id)
            .eq("session_id", self.session_id)
            .gte("created_at", cutoff.isoformat())
            .order("created_at", desc=False)
            .limit(self.max_messages)
            .execute()
        )
        rows = resp.data or []
        return [_to_lc_message(r["role"], r["content"]) for r in rows]