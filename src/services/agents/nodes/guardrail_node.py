import logging
import re
import time
from typing import Dict, Literal

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GuardrailScoring
from ..prompts import GUARDRAIL_PROMPT
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)

# Rule-based patterns for fast rejection/acceptance (no LLM needed)
_OUT_OF_SCOPE_PATTERNS = [
    r"^\b(hi|hello|hey|good\s*(morning|afternoon|evening)|bye|thanks|thank you)\b",
    r"^(what is|who is|define|meaning of)\s+(a |an |the )?\w+\s*$",
    r"^(tell me about|explain)\s+(the meaning of|what is)\s",
    r"^\d+\s*(\+|\-|\*|\/)\s*\d+",
    r"^(write|create|generate)\s+(a )?(poem|story|essay|letter)",
    r"^(recipe|cooking|baking)\s",
    r"^(weather|news|sports|score)\s",
]

_IN_SCOPE_PATTERNS = [
    r"(transformer|attention|bert|gpt|llm|neural|deep\s*learning|machine\s*learning)",
    r"(reinforcement\s*learning|gradient|optimizer|loss\s*function)",
    r"(convolution|recurrent|lstm|gru|gan|autoencoder|vae)",
    r"(natural\s*language|nlp|computer\s*vision|object\s*detection|segmentation)",
    r"(retrieval|embedding|vector|semantic|similarity|rank)",
    r"(paper|arxiv|research|study|experiment|benchmark|evaluation)",
    r"(neural\s*network|activation\s*function|batch\s*normalization|dropout)",
    r"(fine[\s-]tuning|pre[\s-]training|transfer\s*learning|few[\s-]shot)",
]


def _rule_based_check(query: str) -> tuple[int, str, str] | None:
    """Fast rule-based check for obvious cases.

    Returns None if the query needs LLM evaluation.
    Returns (score, reason, query_type) for obvious cases.
    """
    query_lower = query.lower().strip()

    # Check out-of-scope patterns
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if re.match(pattern, query_lower):
            return (20, f"Rule-based rejection: matches non-research pattern", "out_of_scope")

    # Check in-scope patterns
    for pattern in _IN_SCOPE_PATTERNS:
        if re.search(pattern, query_lower):
            return (85, f"Rule-based acceptance: matches research topic", "local_papers")

    return None


def continue_after_guardrail(state: AgentState, runtime: Runtime[Context]) -> Literal["continue", "out_of_scope"]:
    """Determine whether to continue or reject based on guardrail results.

    This function checks the guardrail_result score against a threshold.
    If the score is above threshold, continue; otherwise route to out_of_scope.

    :param state: Current agent state with guardrail results
    :param runtime: Runtime context containing guardrail threshold
    :returns: "continue" if score >= threshold, "out_of_scope" otherwise
    """
    guardrail_result = state.get("guardrail_result")
    if not guardrail_result:
        logger.warning("No guardrail result found, defaulting to out_of_scope (fail-closed)")
        return "out_of_scope"

    score = guardrail_result.score
    threshold = runtime.context.guardrail_threshold

    logger.info(f"Guardrail score: {score}, threshold: {threshold}")

    return "continue" if score >= threshold else "out_of_scope"


async def ainvoke_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, GuardrailScoring]:
    """Asynchronously invoke the guardrail validation step using LLM.

    This function evaluates whether the user query is within scope
    (CS/AI/ML research papers) and assigns a score.

    Uses rule-based fast path for obvious cases, LLM for ambiguous queries.
    Fail-open: if LLM fails, uses rule-based fallback instead of rejecting.

    :param state: Current agent state
    :param runtime: Runtime context
    :returns: Dictionary with guardrail_result
    """
    logger.info("NODE: guardrail_validation")
    start_time = time.time()

    query = get_latest_query(state["messages"])
    logger.debug(f"Evaluating query: {query[:100]}...")

    # Create span for guardrail validation
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.create_span(
                trace=runtime.context.trace,
                name="guardrail_validation",
                input_data={
                    "query": query,
                    "threshold": runtime.context.guardrail_threshold,
                },
                metadata={
                    "node": "guardrail",
                    "model": runtime.context.model_name,
                },
            )
            logger.debug("Created Langfuse span for guardrail validation")
        except Exception as e:
            logger.warning(f"Failed to create span for guardrail validation: {e}")

    # Step 1: Try rule-based check first (fast path)
    rule_result = _rule_based_check(query)
    if rule_result is not None:
        score, reason, query_type = rule_result
        response = GuardrailScoring(score=score, reason=reason)
        response.query_type = query_type  # type: ignore[attr-defined]
        logger.info(f"Rule-based guardrail: score={score}, type={query_type}")
    else:
        # Step 2: LLM validation for ambiguous queries
        try:
            guardrail_prompt = GUARDRAIL_PROMPT + "\n\nUser Query:\n" + query

            llm = runtime.context.ollama_client.get_langchain_model(
                model=runtime.context.model_name,
                temperature=0.0,
            )

            logger.info("Invoking LLM for guardrail validation")
            raw_response = await llm.ainvoke(guardrail_prompt)
            raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            logger.debug(f"Raw guardrail response: {raw_text}")

            # Parse JSON safely
            from .utils import parse_json_safely
            parsed_dict = parse_json_safely(raw_text)

            response = GuardrailScoring(
                score=int(parsed_dict.get("score", 60)),
                reason=str(parsed_dict.get("reason", "Parsed from LLM text")),
                query_type=parsed_dict.get("query_type", "local_papers"),
            )
            logger.info(f"LLM guardrail result - Score: {response.score}, Reason: {response.reason}")

        except Exception as e:
            logger.warning(f"LLM guardrail validation failed: {type(e).__name__}: {e}")
            # Fail-open: use rule-based fallback instead of rejecting
            rule_result = _rule_based_check(query)
            if rule_result is not None:
                score, reason, query_type = rule_result
                response = GuardrailScoring(score=score, reason=f"LLM failed, rule-based fallback: {reason}")
                response.query_type = query_type  # type: ignore[attr-defined]
                logger.warning(f"Using rule-based fallback after LLM failure: score={score}")
            else:
                # No rule matches, fail-open with moderate score
                response = GuardrailScoring(
                    score=60,
                    reason="LLM unavailable, fail-open with moderate score",
                )
                response.query_type = "local_papers"  # type: ignore[attr-defined]
                logger.warning("LLM failed and no rules matched, failing open with score=60")

    # Update span with result
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.end_span(
            span,
            output={
                "score": response.score,
                "reason": response.reason,
                "decision": "continue" if response.score >= runtime.context.guardrail_threshold else "out_of_scope",
            },
            metadata={
                "execution_time_ms": execution_time,
                "threshold": runtime.context.guardrail_threshold,
            },
        )

    return {"guardrail_result": response}
