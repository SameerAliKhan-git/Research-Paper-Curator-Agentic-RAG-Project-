import logging
import time
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.schemas.api.ask import AskRequest, AskResponse
from src.schemas.api.websocket import WSAskRequest, WSMessage

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["websocket"])

FAILED_RESPONSE_PHRASES = [
    "couldn't find any relevant",
    "no relevant information",
    "unable to generate answer",
    "no papers found",
    "no relevant documents",
]


def _is_failed_response(answer: str) -> bool:
    if not answer:
        return True
    answer_lower = answer.lower().strip()
    return any(phrase in answer_lower for phrase in FAILED_RESPONSE_PHRASES)


async def _retrieve_chunks(
    request: WSAskRequest,
    opensearch_client,
    embeddings_service,
) -> tuple[List[Dict], List[str]]:
    query_embedding = None

    if request.use_hybrid:
        try:
            query_embedding = await embeddings_service.embed_query(request.query)
        except Exception as e:
            logger.warning(f"Embedding generation failed, falling back to BM25: {e}")

    search_results = await opensearch_client.search_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.top_k,
        from_=0,
        use_hybrid=request.use_hybrid and query_embedding is not None,
        min_score=0.0,
    )

    chunks: List[Dict] = []
    sources_set: set = set()

    for hit in search_results.get("hits", []):
        arxiv_id = hit.get("arxiv_id", "")
        chunks.append(
            {
                "arxiv_id": arxiv_id,
                "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
            }
        )
        if arxiv_id:
            if arxiv_id.startswith("upload_"):
                sources_set.add("#")
            else:
                clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{clean_id}.pdf")

    return chunks, list(sources_set)


async def _build_prompt(query: str, chunks: List[Dict]) -> str:
    from src.services.ollama.prompts import RAGPromptBuilder

    builder = RAGPromptBuilder()
    return builder.create_rag_prompt(query, chunks)


async def _send(ws: WebSocket, msg: WSMessage) -> None:
    await ws.send_text(msg.model_dump_json())


def _make_ask_request(req: WSAskRequest) -> AskRequest:
    return AskRequest(
        query=req.query,
        model=req.model,
        top_k=req.top_k,
        use_hybrid=req.use_hybrid,
    )


async def _send_cached_response(ws: WebSocket, cached: AskResponse) -> None:
    await _send(
        ws,
        WSMessage(
            type="metadata",
            data={
                "sources": cached.sources,
                "chunks_used": cached.chunks_used,
                "search_mode": cached.search_mode,
            },
        ),
    )
    for word in cached.answer.split():
        await _send(ws, WSMessage(type="chunk", data={"text": word + " "}))
    await _send(ws, WSMessage(type="done", data={"answer": cached.answer}))


@ws_router.websocket("/ws/ask")
async def ws_ask(websocket: WebSocket):
    """WebSocket endpoint for streaming RAG answers."""

    await websocket.accept()

    opensearch_client = websocket.app.state.opensearch_client
    embeddings_service = websocket.app.state.embeddings_service
    ollama_client = websocket.app.state.ollama_client
    cache_client = getattr(websocket.app.state, "cache_client", None)
    semantic_cache = getattr(websocket.app.state, "semantic_cache", None)

    try:
        raw = await websocket.receive_text()
        request = WSAskRequest.model_validate_json(raw)
    except Exception as e:
        logger.warning(f"Invalid WS request: {e}")
        await _send(
            websocket,
            WSMessage(type="error", data={"detail": "Invalid request payload"}),
        )
        await websocket.close()
        return

    start = time.time()

    try:
        # 1. Exact cache
        if cache_client:
            try:
                cached = await cache_client.find_cached_response(_make_ask_request(request))
                if cached:
                    await _send_cached_response(websocket, cached)
                    return
            except Exception as e:
                logger.warning(f"Exact cache check failed: {e}")

        # 2. Semantic cache
        query_embedding = None
        if semantic_cache:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                cached = await semantic_cache.find_semantic(query_embedding, _make_ask_request(request))
                if cached:
                    await _send_cached_response(websocket, cached)
                    return
            except Exception as e:
                logger.warning(f"Semantic cache check failed: {e}")

        # 3. Retrieve chunks
        chunks, sources = await _retrieve_chunks(request, opensearch_client, embeddings_service)

        if not chunks:
            await _send(
                websocket,
                WSMessage(
                    type="metadata",
                    data={
                        "sources": [],
                        "chunks_used": 0,
                        "search_mode": "bm25",
                    },
                ),
            )
            await _send(
                websocket,
                WSMessage(
                    type="done",
                    data={"answer": "I couldn't find any relevant information in the papers to answer your question."},
                ),
            )
            return

        search_mode = "hybrid" if request.use_hybrid else "bm25"
        await _send(
            websocket,
            WSMessage(
                type="metadata",
                data={
                    "sources": sources,
                    "chunks_used": len(chunks),
                    "search_mode": search_mode,
                },
            ),
        )

        # 4. Stream LLM response
        full_answer = ""
        async for chunk in ollama_client.generate_rag_answer_stream(query=request.query, chunks=chunks, model=request.model):
            if chunk.get("response"):
                token = chunk["response"]
                full_answer += token
                await _send(websocket, WSMessage(type="chunk", data={"text": token}))
            if chunk.get("done", False):
                break

        await _send(websocket, WSMessage(type="done", data={"answer": full_answer}))

        # 5. Populate caches
        if full_answer and not _is_failed_response(full_answer):
            ask_resp = AskResponse(
                query=request.query,
                answer=full_answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode=search_mode,
            )
            if cache_client:
                try:
                    await cache_client.store_response(_make_ask_request(request), ask_resp)
                except Exception as e:
                    logger.warning(f"Failed to store in exact cache: {e}")
            if semantic_cache:
                try:
                    if query_embedding is None:
                        query_embedding = await embeddings_service.embed_query(request.query)
                    await semantic_cache.store(_make_ask_request(request), ask_resp, query_embedding)
                except Exception as e:
                    logger.warning(f"Failed to store in semantic cache: {e}")

        logger.info(f"WS ask completed in {time.time() - start:.2f}s | query={request.query[:50]}")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Unhandled WS error: {e}")
        try:
            await _send(
                websocket,
                WSMessage(type="error", data={"detail": "Internal server error"}),
            )
        except Exception:
            pass
        finally:
            await websocket.close()
