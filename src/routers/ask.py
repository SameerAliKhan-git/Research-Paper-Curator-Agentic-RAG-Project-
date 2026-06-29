import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from src.database import get_db_session
from src.dependencies import APIKeyDep, CacheDep, EmbeddingsDep, LangfuseDep, OllamaDep, OpenSearchDep, RerankerDep, SemanticCacheDep, TenantDep, WebSearchDep, ConversationMemoryDep
from src.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.repositories.conversation import ConversationRepository
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])

# Responses that indicate no useful results - should NOT be cached
FAILED_RESPONSE_PHRASES = [
    "couldn't find any relevant",
    "no relevant information",
    "unable to generate answer",
    "no papers found",
    "no relevant documents",
]


def _is_failed_response(answer: str) -> bool:
    """Check if a response indicates no useful results were found.

    Failed responses should NOT be cached because:
    1. The failure might be transient (LLM unavailable, search timeout)
    2. New papers might be indexed later
    3. Caching failures blocks legitimate retries
    """
    if not answer:
        return True
    answer_lower = answer.lower().strip()
    return any(phrase in answer_lower for phrase in FAILED_RESPONSE_PHRASES)


async def classify_query_mode_with_llm(query: str, ollama_client, model: str) -> str:
    """Intelligently classify user query into search modes: colpali, web_search, agentic, hybrid."""
    query_lower = query.lower()
    
    # 1. ColPali Vision Search Keywords
    colpali_keywords = [
        "layout", "visual", "page", "image", "figure", "table", "picture", 
        "scanned", "diagram", "chart", "draw", "plot", "pdf page", "screenshot"
    ]
    if any(kw in query_lower for kw in colpali_keywords):
        logger.info(f"Classified query as colpali (visual) via keywords: {query}")
        return "colpali"
        
    # 2. Web Search Keywords
    web_keywords = [
        "current", "latest", "now", "news", "recent", "today", "yesterday", 
        "2024", "2025", "2026", "web", "internet", "google", "weather", 
        "stock", "price", "temporal"
    ]
    if any(kw in query_lower for kw in web_keywords):
        logger.info(f"Classified query as web_search via keywords: {query}")
        return "web_search"
        
    # 3. Agentic Keywords
    agentic_keywords = [
        "compare", "evaluate", "step-by-step", "mathematical", "derive", 
        "critique", "verify", "why", "how", "reason", "deconstruct", 
        "systematic", "validate", "check"
    ]
    if any(kw in query_lower for kw in agentic_keywords):
        logger.info(f"Classified query as agentic via keywords: {query}")
        return "agentic"
        
    # Otherwise, ask the fast local model to classify the intent
    system_prompt = (
        "You are an intelligent query router. Classify the user query into one of the following search modes:\n"
        "- 'colpali': if the user specifically requests visual pages, page layouts, figures, tables, charts, or document structure.\n"
        "- 'web_search': if the query asks about real-time, very recent events, or information not found in static scientific papers.\n"
        "- 'agentic': if the query is complex, requiring logical step-by-step verification, critique, or multi-paper synthesis.\n"
        "- 'hybrid': for standard factual questions about machine learning, transformers, embeddings, and general AI literature.\n\n"
        "Respond with ONLY the mode name in lowercase ('colpali', 'web_search', 'agentic', or 'hybrid') and absolutely nothing else."
    )
    try:
        response_data = await ollama_client.generate(
            model=model,
            prompt=f"System: {system_prompt}\nUser Query: {query}\nClassification:"
        )
        response_text = response_data.get("response", "") if response_data else ""
        response_clean = response_text.strip().lower()
        for valid_mode in ["colpali", "web_search", "agentic", "hybrid"]:
            if valid_mode in response_clean:
                logger.info(f"Classified query as {valid_mode} via LLM: {query}")
                return valid_mode
    except Exception as e:
        logger.warning(f"Ollama query classification failed: {e}")
        
    return "hybrid"


async def _rerank_chunks_if_enabled(
    query: str,
    chunks: List[Dict],
    sources: List[str],
    reranker_client,
) -> tuple[List[Dict], List[str]]:
    """Rerank retrieved chunks and sort sources using Cross-Encoder Reranker if enabled."""
    from src.config import get_settings
    settings = get_settings()
    if not settings.enable_reranker or not reranker_client or not chunks:
        return chunks, sources

    logger.info(f"Reranking {len(chunks)} chunks with query: {query}")
    try:
        # Prepare docs for reranking
        docs = []
        for idx, chunk in enumerate(chunks):
            docs.append({
                "chunk_text": chunk["chunk_text"],
                "arxiv_id": chunk["arxiv_id"],
                "original_index": idx
            })
        
        results = await reranker_client.rerank(query=query, documents=docs, top_n=len(docs))
        
        if results:
            reranked_chunks = []
            reranked_sources = []
            seen_sources = set()
            
            for item in results:
                original_doc = item.document
                orig_idx = original_doc["original_index"]
                reranked_chunks.append(chunks[orig_idx])
                
                arxiv_id = original_doc["arxiv_id"]
                if arxiv_id:
                    if arxiv_id.startswith("upload_"):
                        source_url = "#"
                    else:
                        arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                        source_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                    
                    if source_url not in seen_sources:
                        seen_sources.add(source_url)
                        reranked_sources.append(source_url)
            
            for s in sources:
                if s not in seen_sources:
                    reranked_sources.append(s)
                    
            return reranked_chunks, reranked_sources
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Falling back to original order.")
        
    return chunks, sources


async def _prepare_chunks_and_sources(
    request: AskRequest,
    opensearch_client,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
    query_embedding=None,
    tenant_id=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for RAG with clean tracing."""

    # Handle embeddings for hybrid search
    if request.use_hybrid and query_embedding is None:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(span=embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = await opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
            tenant_id=tenant_id,
        )

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        seen_parents = set()
        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # If parent-child chunking is enabled, retrieve the larger parent chunk text
            parent_id = hit.get("parent_id")
            parent_text = hit.get("parent_text")

            if parent_text:
                # Deduplicate overlapping parent chunks to keep context token-efficient
                identifier = parent_id if parent_id else parent_text
                if identifier in seen_parents:
                    continue
                seen_parents.add(identifier)
                chunk_text_to_use = parent_text
            else:
                chunk_text_to_use = hit.get("chunk_text", hit.get("abstract", ""))

            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": chunk_text_to_use,
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                if arxiv_id.startswith("upload_"):
                    sources_set.add("#")
                else:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
    semantic_cache: SemanticCacheDep,
    tenant: TenantDep,
    web_search_service: WebSearchDep,
    conversation_memory: ConversationMemoryDep,
    reranker_client: RerankerDep = None,
) -> AskResponse:
    """Clean RAG endpoint with essential tracing, exact match caching, and semantic caching."""

    # Record the query for popularity tracking
    if cache_client:
        try:
            from src.services.query_tracker import QueryTracker
            tracker = QueryTracker(cache_client.redis)
            await tracker.record_query(request.query)
        except Exception as e:
            logger.warning(f"Failed to record query in ask_question: {e}")

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            # 1. Check exact cache first
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request, tenant_id=tenant.tenant_id)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Exact cache check failed, proceeding: {e}")

            # 2. Check semantic cache if enabled
            query_embedding = None
            if semantic_cache:
                try:
                    query_embedding = await embeddings_service.embed_query(request.query)
                    cached_response = await semantic_cache.find_semantic(query_embedding, request, tenant_id=tenant.tenant_id)
                    if cached_response:
                        logger.info("Returning cached response from semantic cache")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Semantic cache check failed, proceeding: {e}")

            # Determine search strategy intelligently via agent classifier
            search_mode_opt = request.search_mode or "auto"
            if search_mode_opt == "auto":
                resolved_mode = await classify_query_mode_with_llm(request.query, ollama_client, request.model)
            else:
                resolved_mode = search_mode_opt

            logger.info(f"RAG query auto-routed in ask_question to: {resolved_mode}")

            # 1. Handle ColPali Visual Search
            if resolved_mode == "colpali":
                try:
                    from src.services.vision.colpali import ColPaliVisionService
                    colpali_service = ColPaliVisionService(opensearch_client, embeddings_service)
                    hits = await colpali_service.search_visual_pages(
                        query=request.query,
                        top_k=4,
                        tenant_id=tenant.tenant_id,
                    )
                    image_urls = [hit["image_path"] for hit in hits]
                    return AskResponse(
                        query=request.query,
                        answer=f"ColPali vision page layout search executed successfully. Matched {len(hits)} layouts.",
                        sources=image_urls,
                        chunks_used=len(hits),
                        search_mode="colpali",
                    )
                except Exception as e:
                    logger.error(f"ColPali ask query failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Visual search failed: {e}")

            # 2. Handle Agentic LangGraph Workflow
            elif resolved_mode == "agentic":
                try:
                    from src.services.agents.agentic_rag import AgenticRAGService
                    from src.services.agents.config import GraphConfig
                    agentic_rag = getattr(request.app.state, "agentic_rag", None)
                    if not agentic_rag:
                        agentic_rag = AgenticRAGService(
                            opensearch_client=opensearch_client,
                            ollama_client=ollama_client,
                            embeddings_client=embeddings_service,
                            langfuse_tracer=langfuse_tracer,
                            graph_config=GraphConfig(model=request.model)
                        )

                    result = await agentic_rag.ask(
                        query=request.query,
                        model=request.model,
                        tenant_id=tenant.tenant_id
                    )

                    raw_sources = result.get("sources", [])
                    sources = []
                    for s in raw_sources:
                        if isinstance(s, str):
                            sources.append(s)
                        elif isinstance(s, dict):
                            sources.append(s.get("url") or "#")

                    return AskResponse(
                        query=request.query,
                        answer=result.get("answer", ""),
                        sources=sources,
                        chunks_used=len(sources),
                        search_mode="agentic",
                    )
                except Exception as e:
                    logger.error(f"Agentic ask query failed: {e}. Falling back to hybrid.")
                    resolved_mode = "hybrid"

            # 3. Handle Web Search vs Local Retrieval
            chunks = []
            sources = []
            search_mode = resolved_mode

            if resolved_mode == "web_search":
                try:
                    web_docs = await web_search_service.search(request.query)
                    if web_docs:
                        for doc in web_docs:
                            chunks.append({
                                "arxiv_id": "web",
                                "chunk_text": doc.page_content,
                            })
                            sources.append(doc.metadata.get("source", "#"))
                except Exception as e:
                    logger.error(f"Web search integration failed: {e}")
            else:
                # 4. Local retrieval (BM25 / Hybrid)
                use_hybrid_search = (resolved_mode == "hybrid")
                local_request = request.model_copy(update={"use_hybrid": use_hybrid_search})
                chunks, sources, _ = await _prepare_chunks_and_sources(
                    local_request, opensearch_client, embeddings_service, rag_tracer, trace, query_embedding, tenant_id=tenant.tenant_id
                )
                chunks, sources = await _rerank_chunks_if_enabled(
                    request.query, chunks, sources, reranker_client
                )

                # Fallback to web search if no local chunks found
                if not chunks:
                    logger.info(f"No local chunks found, falling back to web search for: {request.query}")
                    try:
                        web_docs = await web_search_service.search(request.query)
                        if web_docs:
                            for doc in web_docs:
                                chunks.append({
                                    "arxiv_id": "web",
                                    "chunk_text": doc.page_content,
                                })
                                sources.append(doc.metadata.get("source", "#"))
                            search_mode = "web_fallback"
                    except Exception as e:
                        logger.error(f"Web search fallback failed: {e}")

            if not chunks:
                response = AskResponse(
                    query=request.query,
                    answer="I couldn't find any relevant information in either the local repository or the web to answer your question.",
                    sources=[],
                    chunks_used=0,
                    search_mode=search_mode,
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Retrieve conversation history if session_id provided
            conversation_history = []
            if request.session_id and conversation_memory:
                try:
                    conversation_history = await conversation_memory.get_history(request.session_id, limit=10)
                except Exception as e:
                    logger.warning(f"Failed to fetch conversation history from service: {e}")

            # Build prompt
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(request.query, chunks)
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)

                if conversation_history:
                    history_text = "\n".join(
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in conversation_history
                    )
                    final_prompt = f"Previous conversation:\n{history_text}\n\n{final_prompt}"

                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer
            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                rag_response = await ollama_client.generate_rag_answer(query=request.query, chunks=chunks, model=request.model)
                answer = rag_response.get("answer", "Unable to generate answer")
                rag_tracer.end_generation(gen_span, answer, request.model)

            # Store Q&A in conversation history
            if request.session_id and conversation_memory:
                try:
                    await conversation_memory.add_message(request.session_id, "user", request.query)
                    await conversation_memory.add_message(request.session_id, "assistant", answer)
                except Exception as e:
                    logger.warning(f"Failed to store conversation history: {e}")

            # Prepare response
            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode="bm25" if not request.use_hybrid else "hybrid",
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Store response in caches (skip if it's a failed/empty response)
            if not _is_failed_response(answer):
                if cache_client:
                    try:
                        await cache_client.store_response(request, response, tenant_id=tenant.tenant_id)
                    except Exception as e:
                        logger.warning(f"Failed to store response in cache: {e}")

                if semantic_cache:
                    try:
                        if query_embedding is None:
                            query_embedding = await embeddings_service.embed_query(request.query)
                        await semantic_cache.store(request, response, query_embedding, tenant_id=tenant.tenant_id)
                    except Exception as e:
                        logger.warning(f"Failed to store in semantic cache: {e}")
            else:
                logger.debug("Skipping cache store for failed/empty response")

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")

            # Graceful degradation: try stale cache before returning error
            if cache_client:
                try:
                    stale_response = await cache_client.find_stale_response(request, tenant_id=tenant.tenant_id)
                    if stale_response:
                        logger.warning(f"Upstream error ({e}), returning stale cached response")
                        return stale_response
                except Exception as cache_err:
                    logger.warning(f"Stale cache fallback also failed: {cache_err}")

            raise HTTPException(status_code=500, detail="An internal error occurred while processing your request")


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    embeddings_service: EmbeddingsDep,
    ollama_client: OllamaDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
    semantic_cache: SemanticCacheDep,
    tenant: TenantDep,
    web_search_service: WebSearchDep,
    reranker_client: RerankerDep = None,
) -> StreamingResponse:
    """Clean streaming RAG endpoint."""

    # Record the query for popularity tracking
    if cache_client:
        try:
            from src.services.query_tracker import QueryTracker
            tracker = QueryTracker(cache_client.redis)
            await tracker.record_query(request.query)
        except Exception as e:
            logger.warning(f"Failed to record query in ask_question_stream: {e}")

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # 1. Check exact cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request, tenant_id=tenant.tenant_id)
                        if cached_response:
                            logger.info("Returning cached response for exact streaming query match")

                            # Send metadata first (same format as non-cached)
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            # Stream the cached response in chunks
                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            # Send completion signal
                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed, proceeding: {e}")

                # 2. Check semantic cache if enabled
                query_embedding = None
                if semantic_cache:
                    try:
                        query_embedding = await embeddings_service.embed_query(request.query)
                        cached_response = await semantic_cache.find_semantic(query_embedding, request, tenant_id=tenant.tenant_id)
                        if cached_response:
                            logger.info("Returning cached response from semantic cache for streaming query match")

                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Semantic cache check failed, proceeding: {e}")

                # Determine search strategy intelligently via agent classifier
                search_mode_opt = request.search_mode or "auto"
                if search_mode_opt == "auto":
                    resolved_mode = await classify_query_mode_with_llm(request.query, ollama_client, request.model)
                else:
                    resolved_mode = search_mode_opt

                logger.info(f"RAG query auto-routed to search mode: {resolved_mode}")

                # 1. Handle ColPali Visual Search
                if resolved_mode == "colpali":
                    yield f"data: {json.dumps({'step': '👁️ Directing to ColPali layout page retrieval index...'})}\n\n"
                    try:
                        from src.services.vision.colpali import ColPaliVisionService
                        colpali_service = ColPaliVisionService(opensearch_client, embeddings_service)
                        hits = await colpali_service.search_visual_pages(
                            query=request.query,
                            top_k=4,
                            tenant_id=tenant.tenant_id,
                        )
                        visual_results = []
                        for hit in hits:
                            visual_results.append({
                                "arxiv_id": hit["arxiv_id"],
                                "page_number": hit["page_number"],
                                "image_path": hit["image_path"],
                                "score": hit["score"],
                                "page_text": hit["page_text"],
                                "layout_stats": hit["layout_stats"],
                            })
                        yield f"data: {json.dumps({'search_mode': 'colpali', 'visual_results': visual_results, 'answer': 'Here are the visual page layouts matching your query.', 'done': True})}\n\n"
                        return
                    except Exception as e:
                        logger.error(f"ColPali search within routing failed: {e}")
                        yield f"data: {json.dumps({'error': f'Visual search failed: {e}'})}\n\n"
                        return

                # 2. Handle Agentic LangGraph Workflow
                elif resolved_mode == "agentic":
                    yield f"data: {json.dumps({'step': '🧠 Starting LangGraph Multi-Agent reasoning cycle...'})}\n\n"
                    try:
                        from src.services.agents.agentic_rag import AgenticRAGService
                        from src.services.agents.config import GraphConfig
                        agentic_rag = getattr(request.app.state, "agentic_rag", None)
                        if not agentic_rag:
                            agentic_rag = AgenticRAGService(
                                opensearch_client=opensearch_client,
                                ollama_client=ollama_client,
                                embeddings_client=embeddings_service,
                                langfuse_tracer=langfuse_tracer,
                                graph_config=GraphConfig(model=request.model)
                            )

                        result = await agentic_rag.ask(
                            query=request.query,
                            model=request.model,
                            tenant_id=tenant.tenant_id
                        )

                        # Stream reasoning steps
                        for step in result.get("reasoning_steps", []):
                            yield f"data: {json.dumps({'step': step})}\n\n"

                        # Build mapped sources
                        raw_sources = result.get("sources", [])
                        sources = []
                        for s in raw_sources:
                            if isinstance(s, str):
                                sources.append(s)
                            elif isinstance(s, dict):
                                sources.append(s.get("url") or "#")

                        # Send metadata
                        yield f"data: {json.dumps({'search_mode': 'agentic', 'sources': sources, 'chunks_used': len(sources)})}\n\n"

                        # Stream response
                        ans = result.get("answer", "")
                        for word in ans.split():
                            yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"

                        yield f"data: {json.dumps({'answer': ans, 'done': True})}\n\n"
                        return
                    except Exception as e:
                        logger.error(f"Agentic search within routing failed: {e}. Falling back to hybrid.")
                        resolved_mode = "hybrid"

                # 3. Handle Web Search
                chunks = []
                sources = []
                search_mode = resolved_mode

                if resolved_mode == "web_search":
                    yield f"data: {json.dumps({'step': '🌐 Querying Google Search APIs for recent context...'})}\n\n"
                    try:
                        web_docs = await web_search_service.search(request.query)
                        if web_docs:
                            for doc in web_docs:
                                chunks.append({
                                    "arxiv_id": "web",
                                    "chunk_text": doc.page_content,
                                })
                                sources.append(doc.metadata.get("source", "#"))
                    except Exception as e:
                        logger.error(f"Streaming web search integration failed: {e}")
                else:
                    # 4. Local retrieval (BM25 / Hybrid)
                    use_hybrid_search = (resolved_mode == "hybrid")
                    yield f"data: {json.dumps({'step': f'🔎 Retrieving local repository literature (Mode: {resolved_mode.upper()})...'})}\n\n"
                    
                    # Temporarily update request options for local preparation
                    local_request = request.model_copy(update={"use_hybrid": use_hybrid_search})
                    chunks, sources, _ = await _prepare_chunks_and_sources(
                        local_request, opensearch_client, embeddings_service, rag_tracer, trace, query_embedding, tenant_id=tenant.tenant_id
                    )
                    chunks, sources = await _rerank_chunks_if_enabled(
                        request.query, chunks, sources, reranker_client
                    )

                    # Fallback to web search if no local chunks found
                    if not chunks:
                        yield f"data: {json.dumps({'step': '⚠️ No papers found in local DB. Falling back to web search...'})}\n\n"
                        try:
                            web_docs = await web_search_service.search(request.query)
                            if web_docs:
                                for doc in web_docs:
                                    chunks.append({
                                        "arxiv_id": "web",
                                        "chunk_text": doc.page_content,
                                    })
                                    sources.append(doc.metadata.get("source", "#"))
                                search_mode = "web_fallback"
                        except Exception as e:
                            logger.error(f"Streaming web search fallback failed: {e}")

                if not chunks:
                    yield f"data: {json.dumps({'answer': 'I couldn\'t find any relevant information in either the local repository or the web.', 'sources': [], 'done': True})}\n\n"
                    return

                # Send metadata first
                metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
                yield f"data: {json.dumps(metadata_response)}\n\n"

                # Build prompt
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    final_prompt = prompt_builder.create_rag_prompt(request.query, chunks)
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in ollama_client.generate_rag_answer_stream(
                        query=request.query, chunks=chunks, model=request.model
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                # Store response in caches (skip if it's a failed/empty response)
                if full_response and not _is_failed_response(full_response):
                    search_mode = "bm25" if not request.use_hybrid else "hybrid"
                    response_to_cache = AskResponse(
                        query=request.query,
                        answer=full_response,
                        sources=sources,
                        chunks_used=len(chunks),
                        search_mode=search_mode,
                    )
                    if cache_client:
                        try:
                            await cache_client.store_response(request, response_to_cache, tenant_id=tenant.tenant_id)
                        except Exception as e:
                            logger.warning(f"Failed to store streaming response in cache: {e}")
                    if semantic_cache:
                        try:
                            if query_embedding is None:
                                query_embedding = await embeddings_service.embed_query(request.query)
                            await semantic_cache.store(request, response_to_cache, query_embedding, tenant_id=tenant.tenant_id)
                        except Exception as e:
                            logger.warning(f"Failed to store streaming response in semantic cache: {e}")
                else:
                    logger.debug("Skipping cache store for failed/empty streaming response")

            except Exception as e:
                logger.error(f"Streaming error: {e}")

                # Graceful degradation: try stale cache before returning error
                if cache_client:
                    try:
                        stale_response = await cache_client.find_stale_response(request, tenant_id=tenant.tenant_id)
                        if stale_response:
                            logger.warning(f"Upstream error ({e}), returning stale cached response (streaming)")
                            metadata = {
                                "sources": stale_response.sources,
                                "chunks_used": stale_response.chunks_used,
                                "search_mode": stale_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata)}\n\n"
                            for chunk in stale_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"
                            yield f"data: {json.dumps({'answer': stale_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as cache_err:
                        logger.warning(f"Stale cache fallback also failed: {cache_err}")

                yield f"data: {json.dumps({'error': 'An internal error occurred'})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
