"""PDF text extraction utilities."""

from __future__ import annotations

import os

import fitz


def read_pdf_text(pdf_path: str) -> str:
    """Extract plain text from a local PDF file. Returns empty string on failure."""
    if not pdf_path or not os.path.exists(pdf_path):
        return ""

    text_chunks = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            txt = page.get_text("text")
            if txt:
                text_chunks.append(txt)
    finally:
        doc.close()

    return "\n\n".join(text_chunks).strip()
