# app/rag/memory_adapter.py
from __future__ import annotations
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.rag.memory_supabase import SupabaseChatMessageHistory

class ChatMemory:
    """
    Classe simples para gerenciar o histórico de conversas usando Supabase.
    Não herda de BaseMemory pois essa classe não existe mais em langchain_core.
    """
    
    def __init__(self, user_id: str, session_id: str, days_to_keep: int = 7, max_messages: int = 200):
        self.history = SupabaseChatMessageHistory(user_id=user_id, session_id=session_id, days_to_keep=days_to_keep, max_messages=max_messages)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Carrega o histórico de conversas para ser usado no prompt."""
        return {"chat_history": self.history.messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Salva a pergunta e resposta no histórico."""
        if "question" in inputs:
            self.history.add_message(HumanMessage(content=inputs["question"]))
        if "answer" in outputs:
            self.history.add_message(AIMessage(content=outputs["answer"]))

    def clear(self) -> None:
        """Limpa o histórico de conversas."""
        self.history.clear()