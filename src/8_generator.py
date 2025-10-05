# generator.py
import os
import json
from typing import List, Dict, Any

CONFIG_PATH = os.getenv("GEMINI_CONFIG_PATH", "config/gemini.json")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(context_items: List[Dict[str, Any]], question: str) -> str:
    """
    context_items: lista de dicts com {chunk_id, page, char_start, char_end, content}
    """
    context_lines = []
    for c in context_items:
        snippet = c["content"]
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        context_lines.append(
            f"- chunk_id={c['chunk_id']} page={c.get('page')} offsets=({c.get('char_start')},{c.get('char_end')}): {snippet}"
        )
    context = "\n".join(context_lines)
    prompt = f"""
Você é um assistente que responde apenas com base no CONTEXTO fornecido.
Se a resposta não estiver suportada por evidências no CONTEXTO, responda exatamente:
"Não encontrei evidências suficientes nos documentos para responder com confiança."

Responda de forma concisa. Inclua citações no formato JSON conforme o schema:
{{
  "answer": "string",
  "citations": [
    {{ "chunk_id": "number", "page": "number", "char_start": "number", "char_end": "number" }}
  ]
}}

Pergunta: {question}

CONTEXTO:
{context}
"""
    return prompt.strip()

def generate_answer(context_items: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    cfg = load_config()
    primary = cfg["llm"]["primary"]
    max_output_tokens = cfg["llm"]["max_output_tokens"]
    temperature = cfg["llm"]["temperature"]

    prompt = build_prompt(context_items, question)

    provider = primary["provider"]
    model = primary["model"]

    try:
        if provider == "google":
            # pip install google-generativeai
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            resp = genai.GenerativeModel(model).generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens}
            )
            text = resp.text or ""
        else:
            raise NotImplementedError("Somente Google implementado neste stub.")
    except Exception as e:
        # Fallback local (stub)
        text = json.dumps({
            "answer": "Desculpe, resposta de fallback (modelo local não implementado).",
            "citations": []
        }, ensure_ascii=False)

    # Tentar parsear JSON direto
    try:
        parsed = json.loads(text)
        if "answer" in parsed and "citations" in parsed:
            return parsed
    except Exception:
        pass

    # Se vier texto livre, tente extrair bloco JSON com heurística
    import re
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if "answer" in parsed and "citations" in parsed:
                return parsed
        except Exception:
            pass

    # Último recurso: embrulhar resposta como texto puro sem citações
    return {"answer": text.strip(), "citations": []}