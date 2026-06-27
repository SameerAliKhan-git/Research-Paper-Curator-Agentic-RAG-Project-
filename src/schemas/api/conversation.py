from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")


class ConversationHistory(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier")
    messages: List[ConversationMessage] = Field(..., description="List of conversation messages")
