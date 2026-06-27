import logging
import time
from typing import Any, Dict
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import DECOMPOSE_QUERY_PROMPT
from ..state import AgentState
from .utils import get_latest_query, parse_json_safely

logger = logging.getLogger(__name__)


async def ainvoke_decompose_query_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """Decompose complex query into sub-queries for multi-hop retrieval.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary updating sub_queries in AgentState
    """
    logger.info("NODE: decompose_query (multi-hop retrieval planner)")
    start_time = time.time()

    question = get_latest_query(state["messages"])
    
    # Create Langfuse span for decomposition
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="query_decomposition",
                input_data={"query": question},
                metadata={"node": "decompose_query", "model": runtime.context.model_name},
            )
        except Exception as e:
            logger.warning(f"Failed to create span for decompose_query node: {e}")

    try:
        # Format decomposition prompt
        formatted_prompt = DECOMPOSE_QUERY_PROMPT.format(query=question)

        # Get LLM client
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        logger.info("Invoking LLM for query decomposition")
        response = await llm.ainvoke(formatted_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse JSON list of sub-queries
        parsed = parse_json_safely(response_text)
        sub_queries = list(parsed.get("sub_queries", [question]))

        if not sub_queries:
            sub_queries = [question]

        logger.info(f"Decomposed query into {len(sub_queries)} sub-queries: {sub_queries}")
        
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "sub_queries": sub_queries,
                    "sub_queries_count": len(sub_queries),
                    "execution_time_ms": execution_time
                }
            )

        return {"sub_queries": sub_queries}

    except Exception as e:
        logger.error(f"Failed to decompose query: {e}. Falling back to single original query.")
        
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"error": str(e), "sub_queries": [question], "fallback": True}
            )

        return {"sub_queries": [question]}
