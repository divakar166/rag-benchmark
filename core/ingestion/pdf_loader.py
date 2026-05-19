"""
PDF Loader — extracts raw text per page from a PDF file.

Uses pdfplumber (better table/layout handling than pypdf).
Falls back to pypdf if pdfplumber fails on a page.

Returns a list of PageDoc objects with text + metadata.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pypdf
from loguru import logger


@dataclass
class PageDoc:
    """Represents a single page of text extracted from a PDF."""
    text: str
    page_number: int          # 1-indexed
    source: str               # filename
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def is_empty(self) -> bool:
        return len(self.text.strip()) < 20  # skip near-empty pages


def load_pdf(file_path: str | Path) -> list[PageDoc]:
    """
    Load a PDF and return a list of PageDoc objects (one per page).
    Empty / near-empty pages are filtered out.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    source = path.name
    pages: list[PageDoc] = []

    logger.info(f"Loading PDF: {source}")

    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(
                        f"pdfplumber failed on page {i}, falling back to pypdf: {e}"
                    )
                    text = _pypdf_fallback(path, i - 1)  # pypdf is 0-indexed

                doc = PageDoc(text=text.strip(), page_number=i, source=source)
                if not doc.is_empty():
                    pages.append(doc)
                else:
                    logger.debug(f"Skipped empty page {i}")

    except Exception as e:
        logger.error(f"Failed to open PDF with pdfplumber: {e}")
        raise

    total_chars = sum(p.char_count for p in pages)
    logger.info(
        f"Loaded {len(pages)} pages from '{source}' "
        f"({total_chars:,} chars total)"
    )
    return pages


def _pypdf_fallback(path: Path, page_index: int) -> str:
    """Extract text from a single page using pypdf (0-indexed)."""
    try:
        reader = pypdf.PdfReader(str(path))
        if page_index < len(reader.pages):
            return reader.pages[page_index].extract_text() or ""
    except Exception as e:
        logger.error(f"pypdf fallback also failed on page {page_index + 1}: {e}")
    return ""


def get_full_text(pages: list[PageDoc]) -> str:
    """Concatenate all pages into a single string (useful for chunkers)."""
    return "\n\n".join(p.text for p in pages)
