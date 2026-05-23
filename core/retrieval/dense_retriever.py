"""
Dense Retriever — cosine similarity search against a Qdrant collection.

Used by:
  - Naive RAG
  - Semantic chunking
  - Hierarchical RAG — retrieves child chunks, fetches parents
  - HyDE — same retriever, different query vector
"""
from loguru import logger

from config import settings
from core.ingestion.embedder import embed_query
from vectordb.qdrant_client import qdrant_manager


async def retrieve(
    query: str,
    collection_name: str,
    top_k: int = settings.top_k,
    score_threshold: float = settings.score_threshold,
) -> list[dict]:
    """
    Embed the query and search the collection.

    Returns a list of result dicts:
        {
            "text": str,
            "score": float,
            "chunk_index": int,
            "source": str,
            "page": int | None,
            "metadata": dict,
        }
    """
    logger.debug(f"Retrieving for query='{query[:60]}...' from '{collection_name}'")

    query_vector = await embed_query(query)

    results = qdrant_manager.search(
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    top_score = f"{results[0]['score']:.3f}" if results else "N/A"
    logger.debug(f"Retrieved {len(results)} chunks (top score: {top_score})")

    return results