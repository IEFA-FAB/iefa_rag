# app/rag/memory_adapter.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from app.rag.memory_supabase import SupabaseChatMessageHistory

class ChatMemory:
    """
    Classe simples para gerenciar o histórico de conversas usando Supabase.
    """

    def __init__(self, user_id: str, session_id: str, days_to_keep: int = 7, max_messages: int = 200):
        self.history = SupabaseChatMessageHistory(
            user_id=user_id,
            session_id=session_id,
            days_to_keep=days_to_keep,
            max_messages=max_messages
        )

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Carrega o histórico de conversas para ser usado no prompt."""
        return {"chat_history": self.history.messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Salva a pergunta (user) e a resposta (assistant) estruturada.
        Espera, idealmente, em outputs: {"answer": str, "references": [...], "sources": [...]}
        """
        question = inputs.get("question")
        if isinstance(question, str) and question.strip():
            self.history.add_user_question(question)

        answer = outputs.get("answer")
        references = outputs.get("references", [])
        sources = outputs.get("sources", [])

        if isinstance(answer, str) and answer.strip():
            # Salva com JSON completo (answer/references/sources)
            self.history.add_assistant_answer(answer=answer, references=references, sources=sources)

    def clear(self) -> None:
        """Limpa o histórico de conversas."""
        self.history.clear()