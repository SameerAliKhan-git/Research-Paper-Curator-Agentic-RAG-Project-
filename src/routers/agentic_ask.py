import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import AgenticRAGDep, APIKeyDep, CacheDep, EmbeddingsDep, LangfuseDep, SemanticCacheDep, TenantDep
from src.schemas.api.ask import AgenticAskResponse, AskRequest, FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
    embeddings_service: EmbeddingsDep,
    cache_client: CacheDep = None,
    semantic_cache: SemanticCacheDep = None,
    tenant: TenantDep = None,
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Features:
    - Decides if retrieval is needed
    - Grades document relevance
    - Rewrites queries if needed
    - Provides reasoning transparency

    The agent will automatically:
    1. Determine if the question requires research paper retrieval
    2. If needed, search for relevant papers
    3. Grade retrieved documents for relevance
    4. Rewrite the query if documents aren't relevant
    5. Generate an answer with citations

    Args:
        request: Question and parameters
        agentic_rag: Injected agentic RAG service

    Returns:
        Answer with sources and reasoning steps

    Raises:
        HTTPException: If processing fails
    """
    query_embedding = None
    try:
        # Check semantic cache first
        if semantic_cache:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                sem_cached = await semantic_cache.find_semantic(query_embedding, request, tenant_id=tenant.tenant_id)
                if sem_cached:
                    logger.info("Returning semantically cached agentic response")
                    return AgenticAskResponse(
                        query=sem_cached.query,
                        answer=sem_cached.answer,
                        sources=sem_cached.sources,
                        chunks_used=sem_cached.chunks_used,
                        search_mode=sem_cached.search_mode,
                        reasoning_steps=["Semantic cache hit"],
                        retrieval_attempts=0,
                        trace_id=None,
                        rewritten_query=None,
                    )
            except Exception as e:
                logger.warning(f"Semantic cache check failed: {e}")

        result = await agentic_rag.ask(
            query=request.query,
            model=request.model,
            tenant_id=tenant.tenant_id,
        )

        # Convert sources to URLs (strings) to fit Pydantic schema and frontend
        raw_sources = result.get("sources", [])
        mapped_sources = []
        for src in raw_sources:
            if isinstance(src, dict):
                mapped_sources.append(src.get("url") or src.get("source") or "#")
            elif isinstance(src, str):
                mapped_sources.append(src)
            else:
                mapped_sources.append(str(src))

        response = AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=mapped_sources,
            chunks_used=result.get("chunks_used", len(mapped_sources) if mapped_sources else request.top_k),
            search_mode="hybrid" if request.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            trace_id=result.get("trace_id"),
            rewritten_query=result.get("rewritten_query"),
        )

        # Store in semantic cache (only if actual sources were found)
        if semantic_cache and result.get("sources"):
            try:
                from src.schemas.api.ask import AskResponse

                response_to_cache = AskResponse(
                    query=result["query"],
                    answer=result["answer"],
                    sources=mapped_sources,
                    chunks_used=request.top_k,
                    search_mode="hybrid" if request.use_hybrid else "bm25",
                )
                if query_embedding is None:
                    query_embedding = await embeddings_service.embed_query(request.query)
                await semantic_cache.store(request, response_to_cache, query_embedding, tenant_id=tenant.tenant_id)
            except Exception as e:
                logger.warning(f"Failed to store in semantic cache: {e}")

        return response

    except ValueError as e:
        raise HTTPException(status_code=422, detail="Invalid request parameters")
    except Exception as e:
        logger.error(f"Agentic RAG error: {e}")

        # Graceful degradation: try stale cache before returning error
        if cache_client:
            try:
                stale_response = await cache_client.find_stale_response(request, tenant_id=tenant.tenant_id)
                if stale_response:
                    logger.warning(f"Upstream error ({e}), returning stale cached agentic response")
                    return AgenticAskResponse(
                        query=stale_response.query,
                        answer=stale_response.answer,
                        sources=stale_response.sources,
                        chunks_used=stale_response.chunks_used,
                        search_mode=stale_response.search_mode,
                        reasoning_steps=["Degraded: stale cache response"],
                        retrieval_attempts=0,
                        trace_id=None,
                        rewritten_query=None,
                    )
            except Exception as cache_err:
                logger.warning(f"Stale cache fallback also failed: {cache_err}")

        raise HTTPException(status_code=500, detail="An internal error occurred while processing your question")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    langfuse_tracer: LangfuseDep,
) -> FeedbackResponse:
    """
    Submit user feedback for an agentic RAG response.

    This endpoint allows users to rate the quality of answers and provide
    optional comments. Feedback is tracked in Langfuse for continuous improvement.

    Args:
        request: Feedback data including trace_id, score, and optional comment
        langfuse_tracer: Injected Langfuse tracer service

    Returns:
        FeedbackResponse indicating success or failure

    Raises:
        HTTPException: If feedback submission fails
    """
    try:
        if not langfuse_tracer:
            raise HTTPException(status_code=503, detail="Langfuse tracing is disabled. Cannot submit feedback.")

        success = langfuse_tracer.submit_feedback(
            trace_id=request.trace_id,
            score=request.score,
            comment=request.comment,
        )

        if success:
            # Flush to ensure feedback is sent immediately
            langfuse_tracer.flush()

            return FeedbackResponse(success=True, message="Feedback recorded successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to submit feedback to Langfuse")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.post("/ask-agentic-stream")
async def ask_agentic_stream(
    request: AskRequest,
    agentic_rag: AgenticRAGDep,
    embeddings_service: EmbeddingsDep,
    cache_client: CacheDep = None,
    semantic_cache: SemanticCacheDep = None,
    tenant: TenantDep = None,
) -> StreamingResponse:
    """Streaming agentic RAG endpoint. Runs agentic workflow then streams word-by-word."""

    async def generate_stream():
        query_embedding = None
        try:
            # Check semantic cache
            if semantic_cache:
                try:
                    query_embedding = await embeddings_service.embed_query(request.query)
                    sem_cached = await semantic_cache.find_semantic(query_embedding, request, tenant_id=tenant.tenant_id)
                    if sem_cached:
                        logger.info("Returning semantically cached agentic response (streaming)")
                        metadata = {
                            "sources": sem_cached.sources,
                            "chunks_used": sem_cached.chunks_used,
                            "search_mode": sem_cached.search_mode,
                            "reasoning_steps": ["Semantic cache hit"],
                        }
                        yield f"data: {json.dumps(metadata)}\n\n"
                        for word in sem_cached.answer.split():
                            yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"
                        yield f"data: {json.dumps({'answer': sem_cached.answer, 'done': True})}\n\n"
                        return
                except Exception as e:
                    logger.warning(f"Semantic cache check failed: {e}")

            result = await agentic_rag.ask(query=request.query, model=request.model, tenant_id=tenant.tenant_id)

            answer = result["answer"]
            sources = result.get("sources", [])
            search_mode = "hybrid" if request.use_hybrid else "bm25"

            metadata = {
                "sources": sources,
                "chunks_used": request.top_k,
                "search_mode": search_mode,
                "reasoning_steps": result.get("reasoning_steps", []),
            }
            yield f"data: {json.dumps(metadata)}\n\n"

            for word in answer.split():
                yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"

            yield f"data: {json.dumps({'answer': answer, 'done': True})}\n\n"

            # Store in semantic cache (only if actual sources were found)
            if semantic_cache and sources:
                try:
                    from src.schemas.api.ask import AskResponse

                    response_to_cache = AskResponse(
                        query=request.query,
                        answer=answer,
                        sources=sources,
                        chunks_used=request.top_k,
                        search_mode=search_mode,
                    )
                    if query_embedding is None:
                        query_embedding = await embeddings_service.embed_query(request.query)
                    await semantic_cache.store(request, response_to_cache, query_embedding, tenant_id=tenant.tenant_id)
                except Exception as e:
                    logger.warning(f"Failed to store in semantic cache: {e}")

        except Exception as e:
            logger.error(f"Streaming agentic error: {e}")

            # Graceful degradation: try stale cache
            if cache_client:
                try:
                    stale_response = await cache_client.find_stale_response(request, tenant_id=tenant.tenant_id)
                    if stale_response:
                        logger.warning(f"Upstream error ({e}), returning stale cached agentic response (streaming)")
                        metadata = {
                            "sources": stale_response.sources,
                            "chunks_used": stale_response.chunks_used,
                            "search_mode": stale_response.search_mode,
                            "reasoning_steps": ["Degraded: stale cache response"],
                        }
                        yield f"data: {json.dumps(metadata)}\n\n"
                        for word in stale_response.answer.split():
                            yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"
                        yield f"data: {json.dumps({'answer': stale_response.answer, 'done': True})}\n\n"
                        return
                except Exception as cache_err:
                    logger.warning(f"Stale cache fallback also failed: {cache_err}")

            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
