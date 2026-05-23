"""
Ingestion Pipeline

Orchestrates the full ingestion flow for each strategy:
    PDF → PageDocs → Chunks → Embeddings → Qdrant

Hierarchical is a special case:
    - Two collections: large (parents) + small (children)
    - Children store parent_id references
    - We handle this with a custom ingest path
"""
from pathlib import Path
from typing import Type

from loguru import logger

from config import settings
from core.ingestion.pdf_loader import load_pdf, PageDoc
from core.ingestion.embedder import embed_texts
from core.ingestion.chunkers.base_chunker import BaseChunker, Chunk
from core.ingestion.chunkers.fixed_chunker import FixedChunker
from core.ingestion.chunkers.semantic_chunker import SemanticChunker
from core.ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from vectordb.qdrant_client import qdrant_manager

# Strategy registry
# Maps strategy name → (chunker class, qdrant collection name)
STRATEGY_REGISTRY: dict[str, tuple[Type[BaseChunker], str]] = {
    "naive":        (FixedChunker,    settings.collection_naive),
    "semantic":     (SemanticChunker, settings.collection_semantic),
    "hybrid":       (FixedChunker,    settings.collection_hybrid),    # same chunks, different retrieval
    "hyde":         (FixedChunker,    settings.collection_naive),     # reuses naive collection
}

EMBED_BATCH_SIZE = 64


async def _embed_and_store(
    chunks: list[Chunk],
    collection_name: str,
    recreate: bool,
    id_offset: int = 0,
) -> int:
    """Shared helper: embed a list of chunks and upsert into Qdrant."""

    # Create collection
    qdrant_manager.create_collection(
        collection_name=collection_name,
        recreate=recreate,
    )

    # Embed in batches
    all_vectors: list[list[float]] = []
    texts = [c.text for c in chunks]

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start: start + EMBED_BATCH_SIZE]
        vectors = await embed_texts(batch)
        all_vectors.extend(vectors)
        logger.debug(f"  Embedded chunks {start}-{start + len(batch)}")

    # Upsert
    points = [
        {
            "id": id_offset + i,
            "vector": vector,
            "payload": chunk.to_payload(),
        }
        for i, (chunk, vector) in enumerate(zip(chunks, all_vectors))
    ]

    stored = qdrant_manager.upsert_chunks(collection_name, points)
    return stored


async def ingest_pdf(
    file_path: str | Path,
    strategy: str = "naive",
    recreate_collection: bool = False,
) -> dict:
    """
    Full ingestion pipeline for one strategy.
    Hierarchical gets special dual-collection handling.
    """
    path = Path(file_path)
    logger.info(f" Ingestion START  strategy={strategy}, file={path.name}")

    # Load PDF 
    pages: list[PageDoc] = load_pdf(path)
    logger.info(f"Step 1 | Loaded {len(pages)} pages")

    # Hierarchical: special dual-collection path 
    if strategy == "hierarchical":
        return await _ingest_hierarchical(pages, path.name, recreate_collection)

    # Standard path (naive / semantic / hybrid / hyde) 
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys()) + ['hierarchical']}"
        )

    # hyde reuses naive's collection — no separate ingestion needed
    if strategy == "hyde":
        logger.info(
            "HyDE reuses the naive collection — "
            "make sure 'naive' has been ingested first."
        )
        return {
            "strategy": "hyde",
            "collection": settings.collection_naive,
            "source": path.name,
            "note": "HyDE reuses naive collection. Ingest 'naive' first.",
        }

    chunker_class, collection_name = STRATEGY_REGISTRY[strategy]

    # Chunk 
    chunker: BaseChunker = chunker_class()
    chunks: list[Chunk] = await chunker.chunk(pages)
    logger.info(f"Step 2 | Created {len(chunks)} chunks")

    # Embed + Store 
    logger.info(f"Step 3 | Embedding + storing {len(chunks)} chunks...")
    stored = await _embed_and_store(chunks, collection_name, recreate=recreate_collection)
    logger.info(f"Step 3 | Stored {stored} points in '{collection_name}'")

    summary = {
        "strategy": strategy,
        "collection": collection_name,
        "source": path.name,
        "pages_loaded": len(pages),
        "chunks_created": len(chunks),
        "vectors_stored": stored,
    }
    logger.info(f" Ingestion DONE  {summary}")
    return summary


async def _ingest_hierarchical(
    pages: list[PageDoc],
    source_name: str,
    recreate: bool,
) -> dict:
    """
    Hierarchical ingestion — two collections in one pass.

    parent collection (rag_hierarchical_large):
        - Large chunks (~1024 tok), IDs: 0..N-1
        - Stored but NOT searched
    child collection (rag_hierarchical_small):
        - Small chunks (~256 tok), IDs: N..N+M-1
        - Searched; each has parent_id pointing to parent chunk_index
    """
    chunker = HierarchicalChunker()

    # Step 1: produce parents
    parents = await chunker.chunk(pages)
    logger.info(f"Step 2 | Created {len(parents)} parent chunks")

    # Step 2: produce children (IDs start after parents)
    children = chunker.chunk_children(parents)
    logger.info(f"Step 2 | Created {len(children)} child chunks")

    # Step 3: embed + store parents
    logger.info("Step 3a | Storing parent chunks...")
    parents_stored = await _embed_and_store(
        parents,
        settings.collection_hierarchical_large,
        recreate=recreate,
        id_offset=0,
    )
    logger.info(f"Step 3a | Stored {parents_stored} parents")

    # Step 4: embed + store children
    logger.info("Step 3b | Storing child chunks...")
    children_stored = await _embed_and_store(
        children,
        settings.collection_hierarchical_small,
        recreate=recreate,
        id_offset=len(parents),  # ensures unique IDs
    )
    logger.info(f"Step 3b | Stored {children_stored} children")

    summary = {
        "strategy": "hierarchical",
        "collections": {
            "large": settings.collection_hierarchical_large,
            "small": settings.collection_hierarchical_small,
        },
        "source": source_name,
        "pages_loaded": len(pages),
        "parent_chunks": parents_stored,
        "child_chunks": children_stored,
        "vectors_stored": parents_stored + children_stored,
    }
    logger.info(f" Hierarchical Ingestion DONE  {summary}")
    return summary


async def ingest_all_strategies(
    file_path: str | Path,
    recreate: bool = False,
) -> list[dict]:
    """Run ingestion for ALL strategies sequentially."""
    all_strategies = list(STRATEGY_REGISTRY.keys()) + ["hierarchical"]

    summaries = []
    for strategy in all_strategies:
        summary = await ingest_pdf(file_path, strategy=strategy, recreate_collection=recreate)
        summaries.append(summary)
    return summaries