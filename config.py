from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
import os


class Settings(BaseSettings):
    llm_provider: Literal["openai", "custom"] = Field(default="custom", alias="LLM_PROVIDER")
    llm_api_key: str = Field(default="dummy", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="http://localhost:8002/v1", alias="LLM_BASE_URL")
    llm_model: str = Field(default="Qwen/Qwen2.5-Coder-7B-Instruct", alias="LLM_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    hf_token: str = Field(default="", env="HF_TOKEN")

    # Ollama (Embeddings) 
    embedding_host: str = Field(default="http://localhost:11434", env="EMBEDDING_HOST")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", env="EMBEDDING_MODEL")
    embedding_dimension: int = 384

    # Qdrant 
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")

    # Collection names per strategy
    collection_naive: str = "rag_naive"
    collection_semantic: str = "rag_semantic"
    collection_hierarchical_small: str = "rag_hierarchical_small"
    collection_hierarchical_large: str = "rag_hierarchical_large"
    collection_hybrid: str = "rag_hybrid"

    # Chunking 
    chunk_size: int = 512
    chunk_overlap: int = 64
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256
    semantic_breakpoint_threshold: float = 0.85

    # Retrieval 
    top_k: int = 5
    score_threshold: float = 0.0
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # Generation 
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.7

    # App 
    log_level: str = "INFO"
    data_dir: str = "data"
    upload_dir: str = "data/uploads"
    environment: Literal["development", "production"] = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.data_dir, exist_ok=True)