import os
from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Ex.: bons modelos (verifique a lista disponível para sua conta)
# Chat: "meta/llama3-70b-instruct" ou "mixtral_8x7b"
# Embeddings: "NV-Embed-QA" ou "nvolveqa_40k"
LLM_MODEL = "meta/llama3-70b-instruct"
EMB_MODEL = os.getenv("EMB_MODEL", "nvidia/nv-embedqa-e5-v5")

llm = ChatNVIDIA(model=LLM_MODEL)  # usa NVIDIA_API_KEY do ambiente
print("LLM OK:", llm.invoke("Diga 'Olá' em 3 palavras.").content)

emb = NVIDIAEmbeddings(model=EMB_MODEL)
v = emb.embed_query("Teste de dimensão dos embeddings.")
print("Embedding dim:", len(v))