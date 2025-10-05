# ocr.py
import os
from typing import Tuple, List, Dict, Any

from docling.pipeline.standard import StandardPipeline
from docling.models import ExportFormat
from chunking import custom_hybrid_chunker

# pipeline: default-heron conforme seu JSON
pipeline = StandardPipeline(pipeline_name="default-heron")

def parse_doc(path: str) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Retorna:
      - md: Markdown completo do doc
      - dj: JSON exportado pelo Docling (metadata/estrutura)
      - chunks: lista de dicts [{content, page_num, char_start, char_end, lang, headings, flags...}]
    """
    res = pipeline.run(path)  # Docling aciona RapidOCR auto se necessário
    md = res.document.export(ExportFormat.MARKDOWN)
    dj = res.document.export(ExportFormat.JSON)

    # chunker híbrido (estrutura + alvo de ~500 tokens, overlap 80)
    chunks = custom_hybrid_chunker(
        res.document,
        target_tokens=500,
        overlap_tokens=80,
        split_on_headings=True,
        keep_tables_intact=True,
        attach_captions=True,
    )
    return md, dj, chunks