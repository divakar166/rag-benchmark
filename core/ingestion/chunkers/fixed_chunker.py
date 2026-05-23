"""
Fixed Chunker — Naive RAG Baseline

Splits text into fixed-size token windows with overlap.
This is the simplest possible approach and intentionally the weakest —
it exists so all other strategies have a baseline to beat.

Why it's weak:
  - Splits on token count with no semantic awareness
  - A chunk can start mid-sentence or mid-thought
  - Overlap helps a bit but doesn't solve the coherence problem
"""
from loguru import logger

from config import settings
from core.ingestion.pdf_loader import PageDoc
from core.ingestion.chunkers.base_chunker import BaseChunker, Chunk


class FixedChunker(BaseChunker):
    """
    Splits each page's text into overlapping fixed-size windows.

    Token estimation: 1 token ≈ 4 characters.
    We work in characters internally and convert at boundaries.
    """

    strategy_name = "naive_fixed"

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,        # tokens
        chunk_overlap: int = settings.chunk_overlap,  # tokens
    ):
        self.chunk_size_chars = chunk_size * 4
        self.overlap_chars = chunk_overlap * 4
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            f"FixedChunker initialised — "
            f"chunk_size={chunk_size} tokens, overlap={chunk_overlap} tokens"
        )

    async def chunk(self, pages: list[PageDoc]) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in pages:
            text = page.text
            start = 0

            while start < len(text):
                end = start + self.chunk_size_chars
                chunk_text = text[start:end].strip()

                if len(chunk_text) < 50:
                    # Skip tiny trailing fragments
                    break

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_index=chunk_index,
                        source=page.source,
                        page=page.page_number,
                        strategy=self.strategy_name,
                    )
                )
                chunk_index += 1

                # Move forward by (chunk_size - overlap)
                step = self.chunk_size_chars - self.overlap_chars
                start += step

        logger.info(
            f"FixedChunker produced {len(chunks)} chunks "
            f"from {len(pages)} pages"
        )
        return chunks
