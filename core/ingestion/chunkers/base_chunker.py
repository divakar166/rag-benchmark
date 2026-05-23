"""
Base Chunker — abstract interface every chunking strategy must implement.

This enforces a consistent output shape so retrievers, evaluators,
and the API never care which strategy produced the chunks.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """
    A single text chunk ready to be embedded and stored.

    Attributes:
        text          : The actual chunk text sent to the LLM
        chunk_index   : Position of this chunk in the document
        source        : Filename of the source PDF
        page          : Page number (1-indexed) this chunk came from
        token_count   : Approximate token count (characters / 4)
        strategy      : Which chunking strategy produced this chunk
        parent_id     : For hierarchical chunking — ID of the parent chunk
        metadata      : Any extra key-value pairs (e.g. section heading)
    """
    text: str
    chunk_index: int
    source: str
    page: Optional[int] = None
    token_count: int = field(init=False)
    strategy: str = "base"
    parent_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Rough token estimate: 1 token ≈ 4 characters
        self.token_count = len(self.text) // 4

    def to_payload(self) -> dict:
        """Convert to Qdrant payload dict (everything except the vector)."""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "page": self.page,
            "token_count": self.token_count,
            "strategy": self.strategy,
            "parent_id": self.parent_id,
            **self.metadata,
        }


class BaseChunker(ABC):
    """All chunking strategies inherit from this."""

    strategy_name: str = "base"

    @abstractmethod
    async def chunk(self, pages: list) -> list[Chunk]:
        """
        Takes a list of PageDoc objects and returns a flat list of Chunks.
        Each implementation defines its own chunking logic.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy='{self.strategy_name}')"
