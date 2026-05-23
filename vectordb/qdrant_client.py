"""
Qdrant client wrapper.
Centralises all vector DB operations so strategies never touch qdrant-client directly.
"""
from typing import Optional
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from config import settings


class QdrantManager:
    """
    Thin wrapper around qdrant-client.
    Each RAG strategy gets its own collection so results are fully isolated.
    """

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        logger.info(
            f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}"
        )

    # Collection management 
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = settings.embedding_dimension,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
        recreate: bool = False,
    ) -> None:
        """Create a collection. If recreate=True, drops existing first."""
        if recreate:
            self.delete_collection(collection_name)

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )
            logger.info(f"Collection '{collection_name}' created.")
        except UnexpectedResponse as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection '{collection_name}' already exists — skipping.")
            else:
                raise

    def delete_collection(self, collection_name: str) -> None:
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted.")
        except Exception:
            pass

    def collection_exists(self, collection_name: str) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def get_collection_info(self, collection_name: str) -> dict:
        info = self.client.get_collection(collection_name)
        
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        }

    # Write operations
    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
    ) -> int:
        """
        Upsert a list of chunk dicts into a collection.

        Each chunk dict must have:
            - id (str | int)
            - vector (list[float])
            - payload (dict)  ← stores text, metadata, chunk_index, etc.
        """
        points = [
            qmodels.PointStruct(
                id=chunk["id"],
                vector=chunk["vector"],
                payload=chunk["payload"],
            )
            for chunk in chunks
        ]

        self.client.upsert(collection_name=collection_name, points=points, wait=True)
        logger.info(f"Upserted {len(points)} points into '{collection_name}'")
        return len(points)

    # Read / Search operations
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = settings.top_k,
        score_threshold: float = settings.score_threshold,
        filter: Optional[qmodels.Filter] = None,
    ) -> list[dict]:
        """
        Dense vector search. Returns list of dicts with text + metadata + score.
        """
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=filter,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "score": hit.score,
                "text": (hit.payload or {}).get("text", ""),
                "chunk_index": (hit.payload or {}).get("chunk_index"),
                "source": (hit.payload or {}).get("source", ""),
                "page": (hit.payload or {}).get("page"),
                "metadata": {
                    k: v
                    for k, v in (hit.payload or {}).items()
                    if k not in ("text", "chunk_index", "source", "page")
                },
            }
            for hit in response.points
        ]

    def fetch_by_ids(
        self, collection_name: str, ids: list[int | str]
    ) -> list[dict]:
        """Fetch points by IDs (used for parent-child retrieval)."""
        results = self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "text": r.payload.get("text", ""),
                "chunk_index": r.payload.get("chunk_index"),
                "source": r.payload.get("source", ""),
                "page": r.payload.get("page"),
            }
            for r in results
        ]


# Singleton
qdrant_manager = QdrantManager()
