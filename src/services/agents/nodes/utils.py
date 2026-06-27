import json
import logging
import re
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)


def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
    """Extract sources from tool messages in conversation.

    :param messages: List of messages from graph state
    :returns: List of SourceItem objects
    """
    sources = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "name"):
            if msg.name == "retrieve_papers":
                # Parse tool response for sources
                # This would need to parse the actual document metadata
                # For now, return empty list
                pass

    return sources


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Extract tool artifacts from messages.

    :param messages: List of messages from graph state
    :returns: List of ToolArtefact objects
    """
    artefacts = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            artefact = ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
            artefacts.append(artefact)

    return artefacts


def create_reasoning_step(
    step_name: str,
    description: str,
    metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Create a reasoning step record.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional metadata
    :returns: ReasoningStep object
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )


def filter_messages(messages: List) -> List[AIMessage | HumanMessage]:
    """Filter messages to include only HumanMessage and AIMessage types.

    Excludes tool messages and other internal message types.

    :param messages: List of messages to filter
    :returns: Filtered list of messages
    """
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> str:
    """Get the latest context from tool messages.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, list):
                formatted_docs = []
                for doc in content:
                    if hasattr(doc, "page_content"):
                        formatted_docs.append(doc.page_content)
                    elif isinstance(doc, dict) and "page_content" in doc:
                        formatted_docs.append(doc["page_content"])
                    elif isinstance(doc, dict) and "chunk_text" in doc:
                        formatted_docs.append(doc["chunk_text"])
                    else:
                        formatted_docs.append(str(doc))
                return "\n\n".join(formatted_docs)
            return str(content) if content is not None else ""

    return ""


def parse_json_safely(text: str) -> dict:
    """Extract and parse a JSON object from text content using regex boundary matches.

    Allows tolerant parsing for small models which wrap JSON in markdown blocks
    or include leading/trailing text.
    """
    if not text:
        raise ValueError("Empty response text")

    # Clean leading/trailing markdown code blocks if present
    text_clean = text.strip()
    if text_clean.startswith("```json"):
        text_clean = text_clean[7:]
    elif text_clean.startswith("```"):
        text_clean = text_clean[3:]
    if text_clean.endswith("```"):
        text_clean = text_clean[:-3]
    text_clean = text_clean.strip()

    # Try standard json loads first
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # Regex search for outer brackets
    match = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if not match:
        raise ValueError("No JSON object bounds found in text")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON substring: {match.group(0)}. Error: {e}")
        raise
