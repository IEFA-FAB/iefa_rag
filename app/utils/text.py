# app/utils/text.py
from __future__ import annotations
import json
from typing import Any, Dict
from unidecode import unidecode


def normalize_text_pt(text: str) -> str:
    s = (text or "").strip().lower()
    try:
        s = unidecode(s)
    except Exception:
        pass
    s = " ".join(s.split())
    return s


def tokenize_pt(text: str) -> list[str]:
    return normalize_text_pt(text).split()


def coerce_metadata(meta: Any) -> Dict[str, Any]:
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str) and meta:
        try:
            return json.loads(meta)
        except Exception:
            return {"raw_metadata": meta}
    return {}