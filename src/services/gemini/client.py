import asyncio
import json
import logging
import time
import os
from typing import Any, Dict, List, Optional, AsyncIterator

import httpx
from src.config import Settings
from src.exceptions import OllamaException # We can raise standard LLM exception or subclasses
from src.schemas.ollama import RAGResponse
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser
from src.services.llm_protocol import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

class GeminiClient(LLMClient):
    """Client for interacting with Google Gemini API using OpenAI-compatibility or direct JSON API."""

    def __init__(self, settings: Settings):
        """Initialize Gemini client with settings."""
        self.api_key = os.getenv("GEMINI_API_KEY", "") or getattr(settings, "gemini_api_key", "")
        self.default_model = "gemini-2.5-flash"
        self.timeout = httpx.Timeout(60.0)
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a shared httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def get_langchain_model(self, model: str, temperature: float = 0.0) -> Any:
        """Get a LangChain-compatible model instance using langchain-openai's ChatOpenAI

        pointing to Google Gemini's OpenAI compatibility endpoint.
        """
        from langchain_openai import ChatOpenAI

        # Use Google Gemini's OpenAI compatibility endpoint
        api_key = self.api_key or "mock_key"
        model_name = model if "gemini" in model else "gemini-2.5-flash"
        
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model=model_name,
            temperature=temperature,
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check if Gemini API is responding by listing models."""
        try:
            models = await self.list_models()
            return {
                "status": "healthy",
                "message": "Gemini API is reachable",
                "models_available": len(models) > 0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Gemini API check failed: {e}"
            }

    async def list_models(self) -> List[str]:
        """List available Gemini models."""
        if not self.api_key:
            return ["gemini-2.5-flash", "gemini-2.5-pro"]
        try:
            client = await self._get_client()
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return [m["name"].split("/")[-1] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
            return ["gemini-2.5-flash", "gemini-2.5-pro"]
        except Exception:
            return ["gemini-2.5-flash", "gemini-2.5-pro"]

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate text using direct Gemini API."""
        start_time = time.time()
        client = await self._get_client()
        model_name = model if "gemini" in model else self.default_model
        
        # Prepare body
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
        body = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        # Support format constraints if passed (e.g. structured output schema)
        if "format" in kwargs and kwargs["format"]:
            # Gemini structured json output
            body["generationConfig"]["responseMimeType"] = "application/json"
            if isinstance(kwargs["format"], dict):
                body["generationConfig"]["responseSchema"] = kwargs["format"]

        response = await client.post(url, json=body)
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code != 200:
            raise Exception(f"Gemini API error ({response.status_code}): {response.text}")
            
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise Exception("No candidates returned from Gemini API")
            
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        # Extract token usage
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)
        
        return LLMResponse(
            text=text,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            metadata={"finish_reason": candidates[0].get("finishReason")}
        )

    async def generate_stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream response chunks from Gemini API using Server-Sent Events (SSE)."""
        client = await self._get_client()
        model_name = model if "gemini" in model else self.default_model
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={self.api_key}"
        body = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        async with client.stream("POST", url, json=body) as response:
            if response.status_code != 200:
                raise Exception(f"Gemini streaming API error ({response.status_code})")
                
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if not data_str:
                        continue
                    try:
                        chunk_data = json.loads(data_str)
                        candidates = chunk_data.get("candidates", [])
                        if candidates:
                            text_chunk = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            # Format to match Ollama's stream chunk payload
                            yield {
                                "response": text_chunk,
                                "done": chunk_data.get("usageMetadata") is not None,
                                "model": model_name
                            }
                    except json.JSONDecodeError:
                        continue

    # RAG helper methods mapping exactly to OllamaClient interface
    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "gemini-2.5-flash",
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate RAG answer using retrieved chunks."""
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)
        
        format_val = None
        if use_structured_output:
            format_val = RAGResponse.model_json_schema()
            
        response = await self.generate(
            prompt=prompt,
            model=model,
            format=format_val
        )
        
        answer_text = response.text
        if use_structured_output:
            return self.response_parser.parse_structured_response(answer_text)
            
        sources = []
        seen_urls = set()
        for chunk in chunks:
            arxiv_id = chunk.get("arxiv_id")
            if arxiv_id:
                if arxiv_id.startswith("upload_"):
                    pdf_url = "#"
                else:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                if pdf_url not in seen_urls:
                    sources.append(pdf_url)
                    seen_urls.add(pdf_url)
                    
        citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))
        
        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": "high",
            "citations": citations[:5],
        }

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "gemini-2.5-flash",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate streaming RAG answer using retrieved chunks."""
        prompt = self.prompt_builder.create_rag_prompt(query, chunks)
        async for chunk in self.generate_stream(prompt=prompt, model=model):
            yield chunk
