"""Simple, efficient Langfuse tracing utility for RAG pipeline."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .client import LangfuseTracer

logger = logging.getLogger(__name__)


class RAGTracer:
    """Clean, purpose-built tracer for RAG operations.

    Wraps the LangfuseTracer client to provide high-level RAG-specific
    tracing context managers. All methods are no-op safe — if Langfuse
    is disabled, tracing calls silently pass through.
    """

    def __init__(self, tracer: LangfuseTracer):
        self.tracer = tracer

    @contextmanager
    def trace_request(self, user_id: str, query: str):
        """Main request trace context manager."""
        span = None
        try:
            span = self.tracer.start_span(
                name="rag_request",
                input_data={"query": query, "user_id": user_id},
                metadata={"session_id": f"session_{user_id}", "simplified_tracing": True},
            )
        except Exception as e:
            logger.warning(f"Failed to start trace_request span: {e}")

        if span is None:
            yield None
        else:
            try:
                with span:
                    yield span
            finally:
                try:
                    self.tracer.flush()
                except Exception:
                    pass

    @contextmanager
    def trace_embedding(self, trace, query: str):
        """Query embedding operation with timing."""
        start_time = time.time()
        span = None
        try:
            span = self.tracer.start_span(
                name="query_embedding",
                input_data={"query": query, "query_length": len(query)},
            )
        except Exception as e:
            logger.warning(f"Failed to start trace_embedding span: {e}")

        if span is None:
            yield None
        else:
            try:
                with span:
                    yield span
                duration = time.time() - start_time
                try:
                    self.tracer.update_span(
                        span=span,
                        output={"embedding_duration_ms": round(duration * 1000, 2), "success": True},
                    )
                except Exception:
                    pass
            except Exception:
                duration = time.time() - start_time
                try:
                    self.tracer.update_span(
                        span=span,
                        output={"embedding_duration_ms": round(duration * 1000, 2), "success": False},
                    )
                except Exception:
                    pass
                raise

    @contextmanager
    def trace_search(self, trace, query: str, top_k: int):
        """Search operation with timing."""
        span = None
        try:
            span = self.tracer.start_span(
                name="search_retrieval",
                input_data={"query": query, "top_k": top_k},
            )
        except Exception as e:
            logger.warning(f"Failed to start trace_search span: {e}")

        if span is None:
            yield None
        else:
            with span:
                yield span

    def end_search(self, span, chunks: List[Dict], arxiv_ids: List[str], total_hits: int):
        """End search span with essential results."""
        if not span:
            return

        try:
            self.tracer.update_span(
                span=span,
                output={
                    "chunks_returned": len(chunks),
                    "unique_papers": len(set(arxiv_ids)),
                    "total_hits": total_hits,
                    "arxiv_ids": list(set(arxiv_ids)),
                },
            )
        except Exception:
            pass

    @contextmanager
    def trace_prompt_construction(self, trace, chunks: List[Dict]):
        """Prompt building with timing."""
        span = None
        try:
            span = self.tracer.start_span(
                name="prompt_construction",
                input_data={"chunk_count": len(chunks)},
            )
        except Exception as e:
            logger.warning(f"Failed to start trace_prompt_construction span: {e}")

        if span is None:
            yield None
        else:
            with span:
                yield span

    def end_prompt(self, span, prompt: str):
        """End prompt span with final prompt."""
        if not span:
            return

        try:
            self.tracer.update_span(
                span=span,
                output={
                    "prompt_length": len(prompt),
                    "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                },
            )
        except Exception:
            pass

    @contextmanager
    def trace_generation(self, trace, model: str, prompt: str):
        """LLM generation with timing."""
        gen = None
        try:
            gen = self.tracer.start_generation(
                name="llm_generation",
                model=model,
                input_data={"prompt_length": len(prompt), "prompt": prompt},
            )
        except Exception as e:
            logger.warning(f"Failed to start trace_generation span: {e}")

        if gen is None:
            yield None
        else:
            with gen:
                yield gen

    def end_generation(self, span, response: str, model: str):
        """End generation span with response."""
        if not span:
            return

        try:
            self.tracer.update_generation(
                generation=span,
                output=response,
                usage_metadata={"response_length": len(response), "model_used": model},
            )
        except Exception:
            pass

    def end_request(self, trace, response: str, total_duration: float):
        """End main request trace."""
        if not trace:
            return

        try:
            self.tracer.update_span(
                span=trace,
                output={
                    "answer": response,
                    "total_duration_seconds": round(total_duration, 3),
                    "response_length": len(response),
                },
            )
        except Exception:
            pass
