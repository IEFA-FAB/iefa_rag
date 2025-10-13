# ocr.py
from typing import Tuple, List, Dict, Any

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# Opcional (recomendado) se quiser controlar o limite de tokens por chunk:
# - Requer instalar transformers: pip install transformers
# - E usar o tokenizer do mesmo modelo de embeddings que você vai usar
# from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
# from transformers import AutoTokenizer

# Conversor único reutilizável (evita re-carregar modelos a cada chamada)
_converter = DocumentConverter()

def parse_doc(path: str) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Converte um documento com Docling e retorna:
      - md: Markdown completo do documento
      - dj: dicionário (JSON) exportado pelo Docling (estrutura/metadata)
      - chunks: lista de dicionários com conteúdo e metadados úteis para RAG

    Observações:
      - Exportações corretas segundo as APIs do Docling:
            • document.export_to_markdown()
            • document.export_to_dict()
      - Chunking híbrido oficial via docling.chunking.HybridChunker.
        Para controlar o tamanho por tokens, use um tokenizer HuggingFace
        (ver bloco comentado abaixo).
    """
    # Converte o arquivo; Docling decide pipeline e OCR automaticamente (RapidOCR/Tesseract se necessário)
    res = _converter.convert(source=path)
    doc = res.document

    # Exportações corretas (sem ExportFormat.*):
    md: str = doc.export_to_markdown()
    dj: Dict[str, Any] = doc.export_to_dict()

    # Chunker híbrido oficial.
    # Simples (sem tokenizer explícito):
    chunker = HybridChunker()

    # Caso queira alvo ~500 tokens por chunk (alinhando com seu exemplo):
    # EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # exemplo
    # tokenizer = HuggingFaceTokenizer(
    #     tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    #     max_tokens=500,  # alvo aprox. de tokens por chunk
    # )
    # chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    chunk_iter = chunker.chunk(dl_doc=doc)

    chunks: List[Dict[str, Any]] = []
    for ch in chunk_iter:
        # Texto enriquecido com contexto (headings/captions) recomendado para embeddings
        enriched_text = chunker.contextualize(chunk=ch)

        # Tenta serializar metadados do chunk (pydantic -> dict), quando disponível
        meta_dict: Dict[str, Any] | None = None
        if hasattr(ch, "meta") and ch.meta is not None:
            meta = ch.meta
            if hasattr(meta, "model_dump"):
                try:
                    meta_dict = meta.model_dump()
                except Exception:
                    meta_dict = None

        # Monta um dicionário de chunk amigável para RAG
        chunk_entry: Dict[str, Any] = {
            "content": enriched_text,      # conteúdo contextualizado (use para embed)
            "raw_text": getattr(ch, "text", None),  # texto “cru” do item
            "meta": meta_dict,             # metadados completos do chunk (quando disponíveis)
        }

        # Se metadados padrão estiverem presentes, expõe alguns campos comuns
        if isinstance(meta_dict, dict):
            # Muitos pipelines incluem ‘headings’/‘captions’ nos metadados do chunk
            chunk_entry["headings"] = meta_dict.get("headings", [])
            chunk_entry["captions"] = meta_dict.get("captions", [])
            # Labels dos itens do documento associados ao chunk (ex.: TEXT, TABLE, PICTURE)
            doc_items = meta_dict.get("doc_items", [])
            chunk_entry["labels"] = [it.get("label") for it in doc_items if isinstance(it, dict)]
            # Página(s), quando disponíveis (para PDF costuma existir; para DOCX, nem sempre)
            # Nome da chave pode variar; mantemos uma tentativa segura:
            for k in ("page_numbers", "pages", "page_nums"):
                if k in meta_dict:
                    chunk_entry["page_numbers"] = meta_dict.get(k)
                    break

        chunks.append(chunk_entry)

    return md, dj, chunks