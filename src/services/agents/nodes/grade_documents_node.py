import logging
import time
from typing import Any, Dict

from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..models import GradeDocuments, GradingResult, SourceItem
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """Grade retrieved documents for relevance using LLM.

    This function uses an LLM to evaluate whether the retrieved documents
    are relevant to the user's query and decides whether to generate an
    answer or rewrite the query for better results.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with routing_decision and grading_results
    """
    logger.info("NODE: grade_documents")
    start_time = time.time()

    # Get query and context
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # Extract document chunks from context for logging
    chunks_preview = []
    if context:
        # Context is a string containing all documents concatenated
        # Let's show a preview of what was retrieved
        context_preview = context[:500] + "..." if len(context) > 500 else context
        chunks_preview = [{"text_preview": context_preview, "length": len(context)}]

    # Create span for document grading
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_grading",
                input_data={
                    "query": question,
                    "context_length": len(context) if context else 0,
                    "has_context": context is not None,
                    "chunks_received": chunks_preview,
                },
                metadata={
                    "node": "grade_documents",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for document grading")
        except Exception as e:
            logger.warning(f"Failed to create span for grade_documents node: {e}")

    if not context:
        logger.warning("No context found, routing to rewrite_query")

        # Update span with no context result
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"routing_decision": "rewrite_query", "reason": "no_context"},
                metadata={"execution_time_ms": execution_time},
            )

        return {"routing_decision": "rewrite_query", "grading_results": []}

    logger.debug(f"Grading context of length {len(context)} characters")

    # Use LLM to grade document relevance
    try:
        # Create grading prompt - safe concatenation (no format injection)
        grading_prompt = GRADE_DOCUMENTS_PROMPT + "\n\nRetrieved Documents:\n" + context + "\n\nUser Question:\n" + question

        # Get LLM from runtime context
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        # Invoke LLM grading
        logger.info("Invoking LLM for document grading")
        raw_response = await llm.ainvoke(grading_prompt)
        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        logger.debug(f"Raw grading response: {raw_text}")

        # Parse JSON safely
        from .utils import parse_json_safely
        parsed_dict = parse_json_safely(raw_text)

        binary_score_val = str(parsed_dict.get("binary_score", "yes")).strip().lower()
        # Coerce to 'yes' or 'no'
        binary_score = "yes" if "yes" in binary_score_val else "no"

        is_relevant = binary_score == "yes"
        score = 1.0 if is_relevant else 0.0

        reasoning = str(parsed_dict.get("reasoning", "Parsed from LLM text"))
        logger.info(f"LLM grading: score={binary_score}, reasoning={reasoning}")

        # Create grading result record
        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=score,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"LLM grading failed: {type(e).__name__}, falling back to heuristic")
        # Lenient fallback heuristic for small LLMs:
        # If we retrieved documents and they have content, assume relevant
        # Small LLMs often fail at structured output parsing
        is_relevant = len(context.strip()) > 50
        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=f"Fallback heuristic (LLM failed): {'documents retrieved and contain content' if is_relevant else 'insufficient content'}",
        )

    # Determine routing
    route = "generate_answer" if is_relevant else "rewrite_query"

    logger.info(f"Grading result: {'relevant' if is_relevant else 'not relevant'}, routing to: {route}")

    # Populate relevant_sources if relevant
    relevant_sources = []
    if is_relevant:
        messages = state.get("messages", [])
        tool_message = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_message = msg
                break

        docs_list = None
        if tool_message:
            if getattr(tool_message, "artifact", None) and isinstance(tool_message.artifact, list):
                docs_list = tool_message.artifact
            elif isinstance(tool_message.content, list):
                docs_list = tool_message.content

        if docs_list:
            for doc in docs_list:
                metadata = {}
                if hasattr(doc, "metadata"):
                    metadata = doc.metadata
                elif isinstance(doc, dict) and "metadata" in doc:
                    metadata = doc["metadata"]

                arxiv_id = metadata.get("arxiv_id", "")
                if not arxiv_id and metadata.get("search_mode") == "web_search":
                    arxiv_id = "web"
                if arxiv_id:
                    url = metadata.get("source", "#")
                    authors_val = metadata.get("authors", [])
                    if isinstance(authors_val, str):
                        authors_val = [authors_val] if authors_val else []

                    relevant_sources.append(
                        SourceItem(
                            arxiv_id=arxiv_id,
                            title=metadata.get("title", "Untitled"),
                            authors=authors_val,
                            url=url,
                            relevance_score=float(metadata.get("score", 0.0)),
                        )
                    )

    # Update span with grading result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "routing_decision": route,
                "is_relevant": is_relevant,
                "score": score,
                "reasoning": grading_result.reasoning,
            },
            metadata={
                "execution_time_ms": execution_time,
                "context_length": len(context),
            },
        )

    return {
        "routing_decision": route,
        "grading_results": [grading_result],
        "relevant_sources": relevant_sources,
    }
