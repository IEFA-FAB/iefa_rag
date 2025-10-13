# chunking.py
from typing import List, Dict, Any
import re

def _approx_token_len(text: str) -> int:
    # Aproximação barata de tokens
    return max(1, len(text.split()))

def _merge_segments(segments: List[Dict[str, Any]], target_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
    out = []
    buf, buf_tokens = [], 0
    for seg in segments:
        tlen = _approx_token_len(seg["content"])
        if buf_tokens + tlen > target_tokens and buf:
            # fecha chunk
            merged = _merge_buf(buf)
            out.append(merged)
            # cria overlap do final do buffer
            if overlap_tokens > 0:
                buf = _overlap_from_end(buf, overlap_tokens)
                buf_tokens = sum(_approx_token_len(s["content"]) for s in buf)
            else:
                buf, buf_tokens = [], 0
        buf.append(seg)
        buf_tokens += tlen
    if buf:
        out.append(_merge_buf(buf))
    return out

def _merge_buf(buf: List[Dict[str, Any]]) -> Dict[str, Any]:
    content = "\n\n".join(s["content"] for s in buf)
    first = buf[0]
    last = buf[-1]
    headings = []
    for s in buf:
        if s.get("headings"):
            headings.extend(h for h in s["headings"] if h not in headings)
    return {
        "content": content,
        "page_num": first.get("page_num"),
        "char_start": first.get("char_start", 0),
        "char_end": last.get("char_end", 0),
        "lang": first.get("lang", "pt"),
        "headings": headings,
        "is_table": any(s.get("is_table") for s in buf),
        "is_code": any(s.get("is_code") for s in buf),
    }

def _overlap_from_end(buf: List[Dict[str, Any]], overlap_tokens: int) -> List[Dict[str, Any]]:
    out = []
    acc = 0
    for s in reversed(buf):
        out.append(s)
        acc += _approx_token_len(s["content"])
        if acc >= overlap_tokens:
            break
    return list(reversed(out))

def custom_hybrid_chunker(docling_document, target_tokens=500, overlap_tokens=80,
                          split_on_headings=True, keep_tables_intact=True, attach_captions=True) -> List[Dict[str, Any]]:
    """
    Extrai segmentos do docling_document preservando estrutura básica (headings, tabelas, código),
    e agrega em chunks de ~target_tokens com overlap.
    Nota: API exata do Docling pode variar; aqui simplificamos acessando elementos como texto plano.
    """
    # 1) Varre blocos/elementos do docling_document e monta "segments"
    # Em produção, percorra a árvore estruturada de Docling (parágrafos, tabelas, figuras, etc.)
    # Aqui: fallback lendo MARKDOWN simplificado do documento
    try:
        md = docling_document.export_plaintext()  # se existir
    except Exception:
        md = docling_document.export_markdown()  # fallback

    lines = [l.strip() for l in md.splitlines()]
    segments = []
    cur_headings = []

    for i, line in enumerate(lines):
        if not line:
            continue
        is_heading = bool(re.match(r"^#{1,6}\s+", line)) if split_on_headings else False
        is_table = line.startswith("|") and line.endswith("|") if keep_tables_intact else False
        is_code = line.startswith("```") or line.startswith("    ")
        seg = {
            "content": line,
            "page_num": None,   # Preencher se Docling fornecer
            "char_start": None, # Preencher se Docling fornecer
            "char_end": None,   # Preencher se Docling fornecer
            "lang": "pt",
            "headings": cur_headings.copy(),
            "is_table": is_table,
            "is_code": is_code,
        }
        if is_heading:
            # atualiza contexto de headings
            htxt = re.sub(r"^#{1,6}\s+", "", line).strip()
            cur_headings = [htxt]
            # headings também são segmentos curtos; opcionalmente pule-os
            continue
        segments.append(seg)

    # 2) Merge por tamanho alvo
    chunks = _merge_segments(segments, target_tokens, overlap_tokens)
    return chunks