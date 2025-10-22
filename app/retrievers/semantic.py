# app/retrievers/semantic.py
from __future__ import annotations
from app.clients import get_vectorstore


def get_semantic_retriever(k: int = 5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})