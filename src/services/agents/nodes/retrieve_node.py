import logging
import time
from typing import Dict, Union

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[int, str, list]]:
    """Initiate retrieval or return fallback if max attempts reached.

    This node creates a tool call to retrieve documents, or returns a fallback
    message if the maximum number of retrieval attempts has been reached.

    :param state: Current agent state
    :param runtime: Runtime context containing max_retrieval_attempts
    :returns: Dictionary with updated state (retrieval_attempts, messages, original_query)
    """
    logger.info("NODE: retrieve")
    start_time = time.time()

    messages = state["messages"]
    question = get_latest_query(messages)
    current_attempts = state.get("retrieval_attempts", 0)

    # Get max attempts from context
    max_attempts = runtime.context.max_retrieval_attempts

    # Store original query if not set
    updates = {}
    if state.get("original_query") is None:
        updates["original_query"] = question
        logger.debug(f"Stored original query: {question[:100]}...")

    # Create span for retrieval initiation
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="document_retrieval_initiation",
                input_data={
                    "query": question,
                    "attempt": current_attempts + 1,
                    "max_attempts": max_attempts,
                },
                metadata={
                    "node": "retrieve",
                    "top_k": runtime.context.top_k,
                },
            )
            logger.debug(f"Created Langfuse span for retrieval attempt {current_attempts + 1}")
        except Exception as e:
            logger.warning(f"Failed to create span for retrieve node: {e}")

    # Check if max attempts reached
    if current_attempts >= max_attempts:
        logger.warning(f"Max retrieval attempts ({max_attempts}) reached")
        fallback_msg = (
            f"I apologize, but I couldn't find relevant research papers after {max_attempts} attempts.\n"
            "This may be because:\n"
            "1. No papers in the database contain relevant information\n"
            "2. The query terms don't match the indexed content\n\n"
            "Please try rephrasing your question with more specific technical terms."
        )

        # Update span with max attempts reached
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"status": "max_attempts_reached", "fallback": True},
                metadata={"execution_time_ms": execution_time},
            )

        return {**updates, "messages": [AIMessage(content=fallback_msg)]}

    # Increment retrieval attempts
    new_attempt_count = current_attempts + 1
    updates["retrieval_attempts"] = new_attempt_count
    logger.info(f"Retrieval attempt {new_attempt_count}/{max_attempts}")

    sub_queries = state.get("sub_queries")
    guardrail_result = state.get("guardrail_result")
    query_type = getattr(guardrail_result, "query_type", "local_papers") if guardrail_result else "local_papers"
    tool_name = "google_search" if query_type == "web_search" else "retrieve_papers"

    if sub_queries and len(sub_queries) > 0:
        logger.info(f"Generating tool calls for decomposed sub-queries: {sub_queries}")
        tool_calls = []
        for idx, q in enumerate(sub_queries):
            tool_calls.append({
                "id": f"retrieve_{new_attempt_count}_{idx}",
                "name": tool_name,
                "args": {"query": q},
            })
        llm_message = AIMessage(content="", tool_calls=tool_calls)
    else:
        # Bind search tools to LLM and let it dynamically select a tool
        try:
            # Get LLM from runtime context
            llm = runtime.context.ollama_client.get_langchain_model(
                model=runtime.context.model_name,
                temperature=0.0,
            )

            from src.services.agents.tools import create_retriever_tool, google_search

            retriever_tool = create_retriever_tool(
                opensearch_client=runtime.context.opensearch_client,
                embeddings_client=runtime.context.embeddings_client,
                top_k=runtime.context.top_k,
                use_hybrid=True,
                tenant_id=runtime.context.tenant_id,
            )
            tools = [retriever_tool, google_search]
            llm_with_tools = llm.bind_tools(tools)

            logger.info("Invoking LLM dynamically with bound search tools")
            llm_message = await llm_with_tools.ainvoke(messages)

            # Verify if llm_message has tool_calls
            if not hasattr(llm_message, "tool_calls") or not llm_message.tool_calls:
                logger.info("LLM did not generate any tool calls. Falling back to query classification.")
                raise ValueError("No tool calls generated")

            # Ensure all tool calls have a unique ID for tracing/tracking
            for tc in llm_message.tool_calls:
                if not tc.get("id"):
                    tc["id"] = f"retrieve_{new_attempt_count}"

        except Exception as e:
            logger.info(f"Ollama dynamic tool binding failed or returned empty: {e}. Falling back to query classification.")
            llm_message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"retrieve_{new_attempt_count}",
                        "name": tool_name,
                        "args": {"query": question},
                    }
                ],
            )

    updates["messages"] = [llm_message]

    logger.debug(f"Created tool call for query: {question[:100]}...")

    # Update span with successful tool call creation
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "status": "tool_call_created",
                "query": question,
                "attempt": new_attempt_count,
            },
            metadata={"execution_time_ms": execution_time},
        )

    return updates


def _normalize_tool_args(tool_args: Union[dict, str, None], default_query: str) -> dict:
    """Normalize tool arguments to ensure a clean string 'query' field is present."""
    if not isinstance(tool_args, dict):
        return {"query": default_query}

    # If query is present and is a non-empty string, use it
    q = tool_args.get("query")
    if isinstance(q, str) and q.strip():
        return {"query": q.strip()}

    # If query is a dictionary, see if we can find a string inside
    if isinstance(q, dict):
        for val in q.values():
            if isinstance(val, str) and len(val.strip()) > 2:
                return {"query": val.strip()}

    # Otherwise check if there are other keys in tool_args that are strings
    for k, val in tool_args.items():
        if isinstance(val, str) and len(val.strip()) > 2 and k != "name":
            return {"query": val.strip()}

    # Fall back to default query
    return {"query": default_query}

def _get_default_query_safe(messages: list) -> str:
    """Safely get the latest query from messages, returning an empty string if not found."""
    try:
        return get_latest_query(messages)
    except ValueError:
        return ""


async def _invoke_tool_directly(tool_obj, query: str):
    """Invoke a LangChain tool directly to retrieve raw Python objects (e.g. list of Documents)

    instead of the serialized string representation returned by tool.ainvoke().
    """
    coroutine = getattr(tool_obj, "coroutine", None)
    if coroutine is not None:
        return await coroutine(query)

    func = getattr(tool_obj, "func", None)
    if func is not None:
        import inspect
        if inspect.iscoroutinefunction(func):
            return await func(query)
        else:
            return func(query)

    return await tool_obj.ainvoke({"query": query})


async def ainvoke_tool_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, list]:
    """Execute search tools dynamically with tenant isolation from runtime context.

    This replaces the static prebuilt ToolNode to support per-request tenant
    isolation inside compiled LangGraph graphs.
    """
    logger.info("NODE: tool_retrieve (dynamic)")
    messages = state["messages"]
    last_message = messages[-1]

    tool_messages = []

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        from src.services.agents.tools import create_retriever_tool, google_search

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            logger.info(f"Executing tool {tool_name!r} dynamically with tenant={runtime.context.tenant_id!r}")

            if tool_name == "retrieve_papers":
                retriever_tool = create_retriever_tool(
                    opensearch_client=runtime.context.opensearch_client,
                    embeddings_client=runtime.context.embeddings_client,
                    top_k=runtime.context.top_k,
                    use_hybrid=True,
                    tenant_id=runtime.context.tenant_id,
                )
                try:
                    clean_args = _normalize_tool_args(tool_args, _get_default_query_safe(messages))
                    result = await _invoke_tool_directly(retriever_tool, clean_args["query"])
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            artifact=result,
                            name=tool_name,
                            tool_call_id=tool_id,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error executing retrieve_papers tool: {e}")
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error executing retrieve_papers tool: {e}",
                            name=tool_name,
                            tool_call_id=tool_id,
                        )
                    )
            elif tool_name == "google_search":
                try:
                    clean_args = _normalize_tool_args(tool_args, _get_default_query_safe(messages))
                    result = await _invoke_tool_directly(google_search, clean_args["query"])
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            artifact=result,
                            name=tool_name,
                            tool_call_id=tool_id,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error executing google_search tool: {e}")
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error executing google_search tool: {e}",
                            name=tool_name,
                            tool_call_id=tool_id,
                        )
                    )
            else:
                logger.warning(f"Requested execution of unknown tool: {tool_name!r}")
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: Unknown tool {tool_name!r}",
                        name=tool_name,
                        tool_call_id=tool_id,
                    )
                )

    return {"messages": tool_messages}
