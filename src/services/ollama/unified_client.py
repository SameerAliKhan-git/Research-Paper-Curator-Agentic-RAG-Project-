import logging
from typing import Any, Dict, List, Optional, AsyncIterator
from src.config import Settings
from src.services.ollama.client import OllamaClient
from src.services.gemini.client import GeminiClient

logger = logging.getLogger(__name__)

class UnifiedLLMClient:
    """Unified LLM Client that dynamically routes queries to Ollama or Gemini

    based on the selected model name or global settings.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ollama_client = OllamaClient(settings)
        self.gemini_client = GeminiClient(settings)

    def _select_client(self, model: str) -> Any:
        """Select either GeminiClient or OllamaClient based on the model name."""
        if model and ("gemini" in model.lower() or "gemma4" in model.lower() or "gemma-4" in model.lower() or "cloud" in model.lower()):
            if self.gemini_client.api_key:
                logger.info(f"UnifiedLLMClient: Routing request to Google Gemini API (model={model})")
                return self.gemini_client
            else:
                logger.warning(f"UnifiedLLMClient: Gemini API requested (model={model}) but GEMINI_API_KEY is not set. Falling back to local Ollama client.")
        
        # Default or fallback to Ollama
        logger.debug(f"UnifiedLLMClient: Routing request to Ollama (model={model})")
        return self.ollama_client

    def get_langchain_model(self, model: str, temperature: float = 0.0) -> Any:
        client = self._select_client(model)
        return client.get_langchain_model(model, temperature)

    async def health_check(self) -> Dict[str, Any]:
        ollama_health = await self.ollama_client.health_check()
        gemini_health = await self.gemini_client.health_check()
        return {
            "status": "healthy" if ollama_health.get("status") == "healthy" or gemini_health.get("status") == "healthy" else "unhealthy",
            "ollama": ollama_health,
            "gemini": gemini_health
        }

    async def list_models(self) -> List[Dict[str, Any]]:
        # Combine models from both clients
        ollama_models = await self.ollama_client.list_models()
        gemini_models = await self.gemini_client.list_models()
        
        # Standardize Gemini list to match Ollama's dict format
        standardized_gemini = [{"name": m, "details": {"parameter_size": "Cloud"}} for m in gemini_models]
        return ollama_models + standardized_gemini

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        client = self._select_client(model)
        # Adapt response if it is GeminiClient (since GeminiClient returns LLMResponse object)
        if client == self.gemini_client:
            res = await client.generate(prompt=prompt, model=model, **kwargs)
            return {
                "response": res.text,
                "model": res.model,
                "prompt_eval_count": res.prompt_tokens,
                "eval_count": res.completion_tokens,
                "total_duration": int(res.latency_ms * 1_000_000)
            }
        else:
            return await client.generate(model=model, prompt=prompt, stream=stream, **kwargs)

    async def generate_stream(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        client = self._select_client(model)
        if client == self.gemini_client:
            async for chunk in client.generate_stream(prompt=prompt, model=model, **kwargs):
                yield chunk
        else:
            async for chunk in client.generate_stream(model=model, prompt=prompt, **kwargs):
                yield chunk

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        client = self._select_client(model)
        return await client.generate_rag_answer(
            query=query,
            chunks=chunks,
            model=model,
            use_structured_output=use_structured_output
        )

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
    ) -> AsyncIterator[Dict[str, Any]]:
        client = self._select_client(model)
        async for chunk in client.generate_rag_answer_stream(
            query=query,
            chunks=chunks,
            model=model
        ):
            yield chunk

    async def close(self):
        await self.ollama_client.close()
        await self.gemini_client.close()
