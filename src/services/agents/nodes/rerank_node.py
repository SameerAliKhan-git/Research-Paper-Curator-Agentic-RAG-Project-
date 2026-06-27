"""Rerank node: re-scores retrieved documents using a cross-encoder reranker."""

import logging
import time
from typing import Dict, List

from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState

logger = logging.getLogger(__name__)


async def ainvoke_rerank_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[ToolMessage]]:
    """Rerank retrieved documents using the cross-encoder reranker.

    This node sits between tool_retrieve and grade_documents. It takes the
    ToolMessage containing retrieved documents, reranks them by relevance
    to the query, and replaces the message content with the reranked order.

    If no reranker is configured or reranking fails, documents pass through
    unchanged (graceful degradation).

    :param state: Current agent state
    :param runtime: Runtime context with reranker_client
    :returns: Dictionary with updated messages (reranked ToolMessage)
    """
    logger.info("NODE: rerank")
    start_time = time.time()

    reranker = runtime.context.reranker_client
    if reranker is None:
        logger.debug("No reranker configured, skipping rerank step")
        return {}

    # Find the last ToolMessage (contains retrieved documents)
    messages = state["messages"]
    tool_message = None
    tool_msg_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], ToolMessage):
            tool_message = messages[i]
            tool_msg_idx = i
            break

    if tool_message is None:
        logger.debug("No ToolMessage found, skipping rerank")
        return {}

    # Extract the query for reranking
    from .utils import get_latest_query

    query = get_latest_query(messages)

    # Parse documents from the ToolMessage content
    content = tool_message.content
    if not content or not isinstance(content, list):
        logger.debug("ToolMessage content is not a list of documents, skipping rerank")
        return {}

    # Convert to reranker format
    docs_for_rerank = []
    for doc in content:
        if isinstance(doc, dict):
            docs_for_rerank.append(doc)
        elif hasattr(doc, "page_content"):
            docs_for_rerank.append(
                {
                    "page_content": doc.page_content,
                    "metadata": getattr(doc, "metadata", {}),
                }
            )

    if not docs_for_rerank:
        logger.debug("No documents to rerank")
        return {}

    # Create Langfuse span
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="rerank_documents",
                input_data={"query": query, "document_count": len(docs_for_rerank)},
                metadata={"node": "rerank", "reranker": type(reranker).__name__},
            )
        except Exception as e:
            logger.warning(f"Failed to create span for rerank node: {e}")

    try:
        results = await reranker.rerank(
            query=query,
            documents=docs_for_rerank,
            top_n=len(docs_for_rerank),
        )

        if results:
            # Reorder documents by reranker scores
            reranked_content = [r.document for r in results]
            reranked_message = ToolMessage(
                content=reranked_content,
                tool_call_id=tool_message.tool_call_id,
            )

            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Reranked {len(docs_for_rerank)} documents in {execution_time:.1f}ms "
                f"(top score: {results[0].relevance_score:.3f})"
            )

            if span:
                runtime.context.langfuse_tracer.end_span(
                    span,
                    output={
                        "reranked_count": len(results),
                        "top_score": results[0].relevance_score,
                        "execution_time_ms": execution_time,
                    },
                )

            # Replace the ToolMessage in the messages list
            updated_messages = list(messages)
            updated_messages[tool_msg_idx] = reranked_message
            return {"messages": updated_messages}

    except Exception as e:
        logger.warning(f"Reranking failed, using original order: {e}")
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"error": str(e), "fallback": "original_order"},
            )

    return {}
