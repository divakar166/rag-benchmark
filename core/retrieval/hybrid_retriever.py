"""
Hybrid Retriever

How it works:
  1. Run BM25 (keyword) search over all stored chunk texts (in-memory index)
  2. Run dense (cosine) search in Qdrant as normal
  3. Fuse both ranked lists using Reciprocal Rank Fusion (RRF)
  4. Return top-k results from the fused ranking

Why it beats Naive:
  - Dense search misses exact keyword matches ("GPT-4", "Section 3.2", names)
  - BM25 misses semantic paraphrases ("car" vs "automobile")
  - RRF fusion gets the best of both without needing score normalisation

Why RRF (not score averaging)?
  - BM25 and cosine scores live on completely different scales
  - RRF only uses *rank positions*, not raw scores → no normalisation needed
  - Formula: RRF(d) = Σ weight / (k + rank(d))  where k=60 is a smoothing constant

In-memory BM25:
  - Scrolls ALL points from Qdrant on first query and builds a BM25Okapi index
  - Index is cached per collection — call invalidate_bm25_cache() after re-ingestion
  - Upgrade path: swap for Qdrant native sparse vectors in Phase 3
"""
from loguru import logger
from rank_bm25 import BM25Okapi

from config import settings
from core.ingestion.embedder import embed_query
from vectordb.qdrant_client import qdrant_manager


# BM25 index cache — keyed by collection name 
_bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def _tokenize(text: str) -> list[str]:
    """Whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def _get_bm25_index(collection_name: str) -> tuple[BM25Okapi, list[dict]]:
    """
    Build (or return cached) BM25 index for a collection.
    Scrolls all Qdrant points and indexes their text payloads.
    """
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    logger.info(f"Building BM25 index for '{collection_name}'...")

    all_chunks: list[dict] = []
    offset = None

    while True:
        results, next_offset = qdrant_manager.client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in results:
            payload = point.payload or {}
            all_chunks.append({
                "id": point.id,
                "text": payload.get("text", ""),
                "chunk_index": payload.get("chunk_index"),
                "source": payload.get("source", ""),
                "page": payload.get("page"),
                "score": 0.0,
                "metadata": {
                    k: v for k, v in payload.items()
                    if k not in ("text", "chunk_index", "source", "page")
                },
            })

        if next_offset is None:
            break
        offset = next_offset

    if not all_chunks:
        raise RuntimeError(f"Collection '{collection_name}' is empty — ingest first.")

    corpus = [_tokenize(c["text"]) for c in all_chunks]
    index = BM25Okapi(corpus)

    _bm25_cache[collection_name] = (index, all_chunks)
    logger.info(f"BM25 index built — {len(all_chunks)} documents indexed")

    return index, all_chunks


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    dense_weight: float = settings.dense_weight,
    sparse_weight: float = settings.sparse_weight,
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ weight / (k + rank)
    Chunks matched by chunk_index (stable identifier across both result sets).
    Returns list sorted by descending RRF score.
    """
    rrf_scores: dict[int, float] = {}
    chunk_map: dict[int, dict] = {}

    # Dense rankings
    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_index"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_weight / (k + rank)
        chunk_map[cid] = chunk

    # BM25 rankings
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_index"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + sparse_weight / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused = []
    for cid in sorted_ids:
        chunk = dict(chunk_map[cid])
        chunk["score"] = round(rrf_scores[cid], 6)
        fused.append(chunk)

    return fused


async def retrieve_hybrid(
    query: str,
    collection_name: str,
    top_k: int = settings.top_k,
    score_threshold: float = settings.score_threshold,
    bm25_candidates: int = 20,
) -> list[dict]:
    """
    Hybrid retrieval: BM25 + dense cosine search fused via RRF.

    Returns same shape as dense_retriever.retrieve():
        [{"text", "score", "chunk_index", "source", "page", "metadata"}, ...]
    """
    logger.debug(f"Hybrid retrieve — query='{query[:60]}...'")

    # Dense retrieval 
    query_vector = await embed_query(query)
    dense_results = qdrant_manager.search(
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k * 2,     # fetch extra — RRF may re-rank all of them
        score_threshold=0.0,  # no pre-filter before fusion
    )

    # BM25 retrieval 
    bm25_index, all_chunks = _get_bm25_index(collection_name)
    bm25_scores = bm25_index.get_scores(_tokenize(query))

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:bm25_candidates]

    bm25_results = []
    for idx in top_bm25_indices:
        chunk = dict(all_chunks[idx])
        chunk["score"] = float(bm25_scores[idx])
        bm25_results.append(chunk)

    # RRF fusion 
    fused = _reciprocal_rank_fusion(dense_results, bm25_results)
    filtered = [c for c in fused if c["score"] >= score_threshold][:top_k]

    logger.debug(
        f"Hybrid — dense={len(dense_results)}, bm25={len(bm25_results)}, "
        f"fused={len(fused)}, returned={len(filtered)}"
    )
    return filtered


def invalidate_bm25_cache(collection_name: str) -> None:
    """
    Drop the cached BM25 index for a collection.
    Call this from main.py after any re-ingestion so the index rebuilds
    on the next query with fresh data.
    """
    if collection_name in _bm25_cache:
        del _bm25_cache[collection_name]
        logger.info(f"BM25 cache invalidated for '{collection_name}'")