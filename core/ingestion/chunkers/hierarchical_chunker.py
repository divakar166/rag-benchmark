"""
Hierarchical Chunker — Strategy 3.

How it works:
  1. Split text into PARENT chunks (large, ~1024 tokens) — these go to the LLM
  2. Split each parent into CHILD chunks (small, ~256 tokens) — these are searched
  3. Each child stores a `parent_id` pointing to its parent's chunk_index
  4. At retrieval time: search children for precision,
     then fetch parent chunks to give the LLM full context

Why it beats Naive:
  - Small child chunks → high retrieval precision (less noise per chunk)
  - Large parent chunks → rich context for generation (full reasoning window)
  - Naive has to compromise between the two; hierarchical doesn't

Storage layout:
  - One collection: `rag_hierarchical_small` (children, searchable)
  - One collection: `rag_hierarchical_large` (parents, fetched by ID)
  - pipeline.py handles BOTH collections via _ingest_hierarchical()

Parent / Child ID assignment:
  - Parents are numbered sequentially: 0, 1, 2, ...
  - Children IDs start after all parents: parent_count + child_index
  - This guarantees no ID collision in either collection
"""
from loguru import logger

from config import settings
from core.ingestion.pdf_loader import PageDoc
from core.ingestion.chunkers.base_chunker import BaseChunker, Chunk


def _split_by_chars(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
    source: str,
    page: int,
    strategy: str,
    start_index: int = 0,
) -> list[Chunk]:
    """Fixed-window character splitter shared by both parent and child passes."""
    chunks = []
    pos = 0
    idx = start_index

    while pos < len(text):
        end = pos + chunk_size_chars
        chunk_text = text[pos:end].strip()

        if len(chunk_text) < 30:
            break

        chunks.append(Chunk(
            text=chunk_text,
            chunk_index=idx,
            source=source,
            page=page,
            strategy=strategy,
        ))
        idx += 1
        pos += chunk_size_chars - overlap_chars

    return chunks


class HierarchicalChunker(BaseChunker):
    """
    Produces two sets of chunks from the same document:
      - parents  : large chunks stored in rag_hierarchical_large
      - children : small chunks stored in rag_hierarchical_small

    pipeline.py usage (special dual-collection path):
        chunker  = HierarchicalChunker()
        parents  = await chunker.chunk(pages)      # Step 1
        children = chunker.chunk_children(parents) # Step 2
    """

    strategy_name = "hierarchical"

    def __init__(
        self,
        parent_chunk_size: int = settings.parent_chunk_size,
        child_chunk_size: int = settings.child_chunk_size,
        overlap_tokens: int = settings.chunk_overlap,
    ):
        self.parent_size_chars = parent_chunk_size * 4
        self.child_size_chars = child_chunk_size * 4
        self.overlap_chars = overlap_tokens * 4

        logger.info(
            f"HierarchicalChunker initialised — "
            f"parent={parent_chunk_size} tok, child={child_chunk_size} tok"
        )

    async def chunk(self, pages: list[PageDoc]) -> list[Chunk]:
        """
        Step 1: produce parent chunks (async to match BaseChunker interface
        used by pipeline.py with `await chunker.chunk(pages)`).
        These are stored in rag_hierarchical_large and sent to the LLM.
        """
        parents: list[Chunk] = []
        idx = 0

        for page in pages:
            page_parents = _split_by_chars(
                text=page.text,
                chunk_size_chars=self.parent_size_chars,
                overlap_chars=self.overlap_chars,
                source=page.source,
                page=page.page_number,
                strategy=self.strategy_name,
                start_index=idx,
            )
            parents.extend(page_parents)
            idx += len(page_parents)

        logger.info(f"HierarchicalChunker produced {len(parents)} parent chunks")
        return parents

    def chunk_children(self, parents: list[Chunk]) -> list[Chunk]:
        """
        Step 2: split each parent into children (sync — no embedding needed here).
        Children are stored in rag_hierarchical_small for dense search.
        Each child's `parent_id` points to the parent's chunk_index so
        main.py can fetch the parent after retrieving a child.

        Child IDs start at len(parents) to avoid collision with parent IDs
        across both Qdrant collections.
        """
        children: list[Chunk] = []
        child_idx = len(parents)  # global ID offset

        for parent in parents:
            pos = 0
            while pos < len(parent.text):
                end = pos + self.child_size_chars
                child_text = parent.text[pos:end].strip()

                if len(child_text) < 30:
                    break

                children.append(Chunk(
                    text=child_text,
                    chunk_index=child_idx,
                    source=parent.source,
                    page=parent.page,
                    strategy=self.strategy_name,
                    parent_id=parent.chunk_index,  # ← key link
                ))
                child_idx += 1
                pos += self.child_size_chars - self.overlap_chars

        logger.info(
            f"HierarchicalChunker produced {len(children)} child chunks "
            f"from {len(parents)} parents"
        )
        return children