from functools import lru_cache

from src.config import get_settings
from src.services.ollama.unified_client import UnifiedLLMClient


@lru_cache(maxsize=1)
def make_ollama_client() -> UnifiedLLMClient:
    """
    Create and return a singleton Unified LLM client instance (supporting Ollama and Gemini).

    Returns:
        UnifiedLLMClient: Configured Unified LLM client
    """
    settings = get_settings()
    return UnifiedLLMClient(settings)

