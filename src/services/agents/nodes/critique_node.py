import logging
import time
from typing import Dict

from langchain_core.messages import ToolMessage
from src.services.agents.state import AgentState
from src.services.agents.models import RoutingDecision
from langgraph.runtime import Runtime
from ..context import Context
from .utils import parse_json_safely

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """You are an expert critique judge. Your task is to evaluate whether the retrieved context contains sufficient, detailed, and relevant information to fully answer the user question.

Read the user question and the retrieved documents, then perform a strict assessment.
If the retrieved documents contain the direct answers, necessary specifications, and contextual details required to address the user question fully, mark it sufficient.
If key aspects, calculations, formulas, context, or definitions are missing, or if the documents are off-topic or only tangentially related, mark it insufficient.

You must respond in JSON format with the following keys:
1. "sufficient": true or false
2. "missing_aspects": a list of specific concepts, details, or questions that are missing from the documents (empty list if sufficient is true)
3. "feedback": a brief explanation of your decision

Example JSON response:
{
  "sufficient": false,
  "missing_aspects": ["specific training duration", "exact validation loss curve details"],
  "feedback": "The retrieved paper mentions the model structure but lacks specific training duration and exact validation loss details requested."
}"""

async def ainvoke_critique_context_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict:
    """Evaluate context sufficiency using LLM-as-a-judge before answer generation.

    If the context is insufficient, routes back to rewrite_query (up to max attempts).
    """
    logger.info("NODE: critique_context")
    start_time = time.time()
    
    # 1. Retrieve the text of all graded documents
    messages = state.get("messages", [])
    question = state.get("original_query", "")
    if not question and messages:
        question = messages[0].content

    # Extract documents from tool message artifacts or content
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

    context_parts = []
    if docs_list:
        for idx, doc in enumerate(docs_list):
            doc_text = getattr(doc, "page_content", "") if hasattr(doc, "page_content") else str(doc)
            if isinstance(doc, dict) and "chunk_text" in doc:
                doc_text = doc["chunk_text"]
            context_parts.append(f"[Doc {idx+1}]: {doc_text}")

    context = "\n\n".join(context_parts) if context_parts else ""

    span = None
    if runtime.context.langfuse_tracer:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="context_critique",
                input_data={
                    "query": question,
                    "context_length": len(context),
                    "document_count": len(context_parts),
                },
                metadata={
                    "node": "critique_context",
                    "model": runtime.context.model_name,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to create span for critique_context node: {e}")

    # Fallback to generate if no context was found (grading would have already handled it, but safety check)
    if not context:
        logger.warning("Critique: Empty context, proceeding to generation or rewrite")
        attempts = state.get("retrieval_attempts", 0)
        route = "rewrite_query" if attempts < runtime.context.max_retrieval_attempts else "generate_answer"
        
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"sufficient": False, "routing_decision": route, "reason": "empty_context"},
            )
        return {"routing_decision": route}

    # Evaluate sufficiency using LLM
    sufficient = True
    feedback = "LLM evaluation skipped or failed"
    missing_aspects = []

    try:
        critique_input = CRITIQUE_PROMPT + "\n\nRetrieved Documents:\n" + context + "\n\nUser Question:\n" + question
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        logger.info("Invoking LLM for context critique evaluation (LLM-as-a-Judge)")
        raw_response = await llm.ainvoke(critique_input)
        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        
        parsed = parse_json_safely(raw_text)
        sufficient = bool(parsed.get("sufficient", True))
        feedback = str(parsed.get("feedback", ""))
        missing_aspects = parsed.get("missing_aspects", [])
        
        logger.info(f"Critique result: sufficient={sufficient}, missing={missing_aspects}")

    except Exception as e:
        logger.error(f"LLM critique failed: {e}, defaulting to sufficient = True to avoid block")
        sufficient = True

    # Check loop limit
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = runtime.context.max_retrieval_attempts if hasattr(runtime.context, "max_retrieval_attempts") else 2

    if not sufficient and attempts < max_attempts:
        route = "rewrite_query"
        logger.warning(f"Critique Judge marked context INSUFFICIENT. Feedback: {feedback}. Routing to query rewrite (attempt {attempts}/{max_attempts})")
        # Add feedback as metadata so the rewriter knows what is missing
        metadata_update = state.get("metadata", {})
        metadata_update["critique_feedback"] = feedback
        metadata_update["missing_aspects"] = missing_aspects
    else:
        route = "generate_answer"
        logger.info("Critique Judge marked context SUFFICIENT. Routing to answer generation.")

    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={"sufficient": sufficient, "feedback": feedback, "routing_decision": route},
            metadata={"execution_time_ms": execution_time},
        )

    return {
        "routing_decision": route,
        "metadata": {
            **(state.get("metadata") or {}),
            "critique_sufficient": sufficient,
            "critique_feedback": feedback,
            "missing_aspects": missing_aspects
        }
    }
