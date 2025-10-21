import os
from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from build_retrievers import get_hybrid_retriever

LLM_MODEL = "meta/llama3-70b-instruct"

def build_chain():
    retriever = get_hybrid_retriever(k_sem=4, k_bm25=6, weights=(0.55, 0.45))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Você é um assistente que responde SOMENTE com base no contexto fornecido. "
             "Se a resposta não estiver no contexto, diga que não encontrou."),
            ("human",
             "Pergunta: {question}\n\n"
             "Contexto:\n{context}\n\n"
             "Responda de forma concisa e cite as fontes (metadata.source) quando possível.")
        ]
    )

    llm = ChatNVIDIA(model=LLM_MODEL)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

if __name__ == "__main__":
    chain, _ = build_chain()
    print(chain.invoke("Resuma os principais tópicos cobertos no(s) PDF(s)."))