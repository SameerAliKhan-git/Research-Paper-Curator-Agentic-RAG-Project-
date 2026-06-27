"""Abstract LLM client protocol for multi-provider support.

Provides a unified interface for LLM operations across different providers
(Ollama, OpenAI, Anthropic, vLLM, etc.). The protocol defines the contract
that all LLM client implementations must follow.

Usage:
    from src.services.llm_protocol import LLMClient, LLMResponse

    class MyCustomClient(LLMClient):
        async def generate(self, prompt: str, model: str, **kwargs) -> LLMResponse:
            ...
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response across providers."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base class for LLM client implementations.

    All LLM providers (Ollama, OpenAI, Anthropic, vLLM, etc.) should
    implement this interface for consistent integration with the RAG pipeline.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt text
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Standardized LLM response
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate text with streaming response.

        Args:
            prompt: Input prompt text
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Streaming response chunks
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM service is healthy and responding.

        Returns:
            Health status dictionary
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
