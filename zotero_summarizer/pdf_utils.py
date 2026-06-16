"""Extract plain text from PDF bytes."""

from __future__ import annotations

import io
from typing import Optional

from pypdf import PdfReader


def extract_text(pdf_bytes: bytes, max_chars: Optional[int] = None) -> str:
    """Return the concatenated text of every page. Truncated to ``max_chars``
    if given so the prompt stays within the model's context window."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            # A single broken page shouldn't abort the whole document.
            continue
    text = "\n".join(parts).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text
