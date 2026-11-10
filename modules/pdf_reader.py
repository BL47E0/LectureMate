"""Optional multimodal extension: read text from lecture slides (PDF)."""

from __future__ import annotations

from typing import BinaryIO

from .utils import LectureMateError

MAX_SLIDE_CHARS = 60_000


def extract_pdf_text(file_obj: BinaryIO, max_chars: int = MAX_SLIDE_CHARS) -> str:
    """Return the text of every page, labelled '[Slide N]'."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise LectureMateError("pypdf is not installed. Run: pip install -r requirements.txt") from exc

    try:
        reader = PdfReader(file_obj)
        if reader.is_encrypted and not reader.decrypt(""):
            raise LectureMateError("The PDF is password-protected and cannot be read.")
        parts: list[str] = []
        total = 0
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            parts.append(f"[Slide {number}]\n{text}")
            total += len(text)
            if total >= max_chars:
                break
    except LectureMateError:
        raise
    except Exception as exc:
        raise LectureMateError(f"Could not read the PDF: {exc}") from exc

    if not parts:
        raise LectureMateError(
            "No selectable text found in the PDF (it may be scanned images). "
            "Continuing without the slides."
        )
    return "\n\n".join(parts)[:max_chars]
