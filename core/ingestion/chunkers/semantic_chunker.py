"""
Semantic Chunker — Strategy 2.

How it works:
  1. Split each page into sentences (using nltk)
  2. Embed ALL sentences in one batched call to embedding-api
  3. Compute cosine similarity between consecutive sentence embeddings
  4. When similarity drops below `breakpoint_threshold` → topic boundary → split here
  5. Accumulate sentences into a chunk until a boundary is hit or
     the chunk exceeds `max_chunk_tokens`

Why it beats Naive:
  - Chunks never cut mid-thought — they end at natural topic transitions
  - Variable size is fine; coherence matters more than uniformity
  - The LLM gets semantically complete context, not arbitrary token windows

Trade-off:
  - Ingestion is slower (sentence embeddings needed upfront)
  - Threshold is a tunable hyperparameter — too high = tiny chunks,
    too low = chunks as large as pages
"""

import numpy as np
import nltk
from loguru import logger

from config import settings
from core.ingestion.pdf_loader import PageDoc
from core.ingestion.chunkers.base_chunker import BaseChunker, Chunk
from core.ingestion.embedder import embed_texts

# Download punkt tokenizer on first use
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using nltk.
    Filters out very short fragments (< 10 chars) that are usually
    headers, page numbers, or noise.
    """
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) >= 10]


class SemanticChunker(BaseChunker):
    """
    Splits text at semantic boundaries detected via embedding similarity drops.

    Key parameters:
        breakpoint_threshold : cosine similarity below this = topic change = split
        max_chunk_tokens     : hard ceiling — never exceed this regardless of similarity
        min_chunk_tokens     : ignore splits that would produce tiny chunks
    """

    strategy_name = "semantic"

    def __init__(
        self,
        breakpoint_threshold: float = settings.semantic_breakpoint_threshold,
        max_chunk_tokens: int = settings.chunk_size,
        min_chunk_tokens: int = 50,
    ):
        self.breakpoint_threshold = breakpoint_threshold
        self.max_chunk_chars = max_chunk_tokens * 4
        self.min_chunk_chars = min_chunk_tokens * 4

        logger.info(
            f"SemanticChunker initialised — "
            f"threshold={breakpoint_threshold}, "
            f"max_chunk={max_chunk_tokens} tokens"
        )

    async def chunk(self, pages: list[PageDoc]) -> list[Chunk]:
        """
        Async — pipeline.py calls this with `await chunker.chunk(pages)`.
        Batches sentence embedding per page to the embedding-api service.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in pages:
            sentences = _split_sentences(page.text)

            if not sentences:
                continue

            if len(sentences) == 1:
                # Single-sentence page — emit as-is
                chunks.append(Chunk(
                    text=sentences[0],
                    chunk_index=chunk_index,
                    source=page.source,
                    page=page.page_number,
                    strategy=self.strategy_name,
                ))
                chunk_index += 1
                continue

            # Embed all sentences in one batched call 
            logger.debug(
                f"Embedding {len(sentences)} sentences from page {page.page_number}"
            )
            embeddings = await embed_texts(sentences)

            # Find breakpoints via similarity drops 
            # similarities[i] = sim between sentence i and sentence i+1
            similarities = [
                _cosine_similarity(embeddings[i], embeddings[i + 1])
                for i in range(len(embeddings) - 1)
            ]

            # Group sentences into chunks 
            current_sentences: list[str] = [sentences[0]]

            for sentence, sim in zip(sentences[1:], similarities):
                current_text = " ".join(current_sentences)
                would_exceed = len(current_text) + len(sentence) > self.max_chunk_chars
                is_breakpoint = sim < self.breakpoint_threshold

                if (is_breakpoint or would_exceed) and len(current_text) >= self.min_chunk_chars:
                    # Emit current chunk at the boundary
                    chunks.append(Chunk(
                        text=current_text.strip(),
                        chunk_index=chunk_index,
                        source=page.source,
                        page=page.page_number,
                        strategy=self.strategy_name,
                        metadata={"breakpoint_similarity": round(sim, 4)},
                    ))
                    chunk_index += 1
                    current_sentences = [sentence]
                else:
                    current_sentences.append(sentence)

            # Emit whatever is left after the last sentence
            if current_sentences:
                final_text = " ".join(current_sentences).strip()
                if len(final_text) >= self.min_chunk_chars:
                    chunks.append(Chunk(
                        text=final_text,
                        chunk_index=chunk_index,
                        source=page.source,
                        page=page.page_number,
                        strategy=self.strategy_name,
                    ))
                    chunk_index += 1

        logger.info(
            f"SemanticChunker produced {len(chunks)} chunks "
            f"from {len(pages)} pages"
        )
        return chunks