import logging
import time
from typing import Any, Dict
from langgraph.runtime import Runtime

from ..context import Context
from ..models import VerificationResult
from ..prompts import VERIFY_ANSWER_PROMPT
from ..state import AgentState
from .utils import get_latest_context, parse_json_safely

logger = logging.getLogger(__name__)


async def ainvoke_verify_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """Verify generated answer against retrieved context using LLM (Hallucination Guard).

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary updating verification_result and is_grounded
    """
    logger.info("NODE: verify_answer (hallucination guard)")
    start_time = time.time()

    # Get latest context and answer
    context = get_latest_context(state["messages"])
    messages = state.get("messages", [])
    
    # Latest generated answer is the last AI message
    answer = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            answer = msg.content
            break

    if not context or not answer:
        logger.info("Context or answer is empty. Skipping verification.")
        return {
            "verification_result": VerificationResult(
                is_grounded=True,
                reasoning="Context or answer empty",
                unsupported_claims=[]
            ),
            "is_grounded": True
        }

    # Create Langfuse span for verification
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="hallucination_verification",
                input_data={"context_len": len(context), "answer_len": len(answer)},
                metadata={"node": "verify_answer", "model": runtime.context.model_name},
            )
        except Exception as e:
            logger.warning(f"Failed to create span for verify_answer node: {e}")

    try:
        # Format prompt
        formatted_prompt = VERIFY_ANSWER_PROMPT.format(context=context[:6000], answer=answer)

        # Get LLM client
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )

        logger.info("Invoking LLM for answer grounding verification")
        response = await llm.ainvoke(formatted_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        parsed = parse_json_safely(response_text)
        is_grounded = bool(parsed.get("is_grounded", True))
        reasoning = str(parsed.get("reasoning", "No reasoning provided"))
        unsupported_claims = list(parsed.get("unsupported_claims", []))

        result = VerificationResult(
            is_grounded=is_grounded,
            reasoning=reasoning,
            unsupported_claims=unsupported_claims
        )

        logger.info(f"Answer verification result: is_grounded={is_grounded}, claims_unsupported={len(unsupported_claims)}")
        
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.end_span(
                span,
                output={
                    "is_grounded": is_grounded,
                    "reasoning": reasoning,
                    "unsupported_claims_count": len(unsupported_claims),
                    "execution_time_ms": execution_time
                }
            )

        return {
            "verification_result": result,
            "is_grounded": is_grounded
        }

    except Exception as e:
        logger.error(f"Failed verifying answer grounding: {e}. Defaulting to grounded=True.")
        
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"error": str(e), "is_grounded": True, "fallback": True}
            )

        return {
            "verification_result": VerificationResult(
                is_grounded=True,
                reasoning=f"Verification failed due to: {str(e)}",
                unsupported_claims=[]
            ),
            "is_grounded": True
        }
