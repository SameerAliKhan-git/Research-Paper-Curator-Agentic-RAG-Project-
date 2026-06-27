from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


class WSMessage(BaseModel):
    """WebSocket message envelope."""

    type: Literal["metadata", "chunk", "done", "error"] = Field(..., description="Message type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message payload")


class WSAskRequest(BaseModel):
    """Request payload sent as the first WebSocket message."""

    query: str = Field(..., min_length=1, max_length=1000)
    model: str = Field("llama3.2:1b")
    top_k: int = Field(3, ge=1, le=10)
    use_hybrid: bool = Field(True)
