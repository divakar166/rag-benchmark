"""
LLM Client Factory - Unified client initialization for custom and OpenAI models.
"""
from openai import AsyncOpenAI
from config import settings

_client = None

def get_llm_client() -> AsyncOpenAI:
    """
    Get or initialize the global AsyncOpenAI client.
    
    Supports:
    1. 'custom' provider: Calls a self-deployed OpenAI-compatible LLM (e.g. vLLM, Ollama)
       using LLM_BASE_URL and LLM_API_KEY.
    2. 'openai' provider: Calls the standard OpenAI API using OPENAI_API_KEY (falls back
       to LLM_API_KEY if not specified) with default base URL.
    """
    global _client
    if _client is None:
        if settings.llm_provider == "openai":
            # For standard OpenAI, do not specify base_url
            api_key = settings.openai_api_key or settings.llm_api_key
            _client = AsyncOpenAI(api_key=api_key)
        else:
            # Custom deployed LLM (vLLM, Ollama, etc.)
            _client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
    return _client
