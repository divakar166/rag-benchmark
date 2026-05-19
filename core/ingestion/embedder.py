"""
Embedder — uses local embedding-api service.
"""

import asyncio
import hashlib
import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from config import settings

_cache: dict[str, list[float]] = {}


def _cache_key(text: str) -> str:
    raw = f"{settings.embedding_model}:{text}"
    return hashlib.md5(raw.encode()).hexdigest()


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _embed_batch(texts: list[str], client: httpx.AsyncClient) -> list[list[float]]:
    response = await client.post(
        f"{settings.embedding_host}/embed",
        json={"texts": texts},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    uncached: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        key = _cache_key(text)

        if key in _cache:
            results[i] = _cache[key]
        else:
            uncached.append((i, text))

    if uncached:
        logger.debug(
            f"Embedding {len(uncached)} texts via embedding-api "
            f"({len(texts) - len(uncached)} cache hits)"
        )

        uncached_texts = [text for _, text in uncached]

        async with httpx.AsyncClient() as client:
            embeddings = await _embed_batch(uncached_texts, client)

        for (i, text), embedding in zip(uncached, embeddings):
            _cache[_cache_key(text)] = embedding
            results[i] = embedding

    if any(r is None for r in results):
        raise RuntimeError("Some embeddings were not generated")

    return results  # type: ignore


async def embed_query(query: str) -> list[float]:
    return (await embed_texts([query]))[0]


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    return asyncio.run(embed_texts(texts))


def embed_query_sync(query: str) -> list[float]:
    return asyncio.run(embed_query(query))