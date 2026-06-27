from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session
from src.models.conversation import Conversation

# Maximum messages per conversation to prevent unbounded JSON blob growth
MAX_MESSAGES_PER_CONVERSATION = 200


class ConversationRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_session(self, session_id: str) -> Optional[Conversation]:
        stmt = select(Conversation).where(Conversation.session_id == session_id)
        return self.session.scalar(stmt)

    def create_or_append(self, session_id: str, role: str, content: str) -> Conversation:
        conversation = self.get_by_session(session_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {"role": role, "content": content, "timestamp": timestamp}

        if conversation:
            conversation.messages.append(entry)
            # Trim oldest messages to prevent unbounded growth
            if len(conversation.messages) > MAX_MESSAGES_PER_CONVERSATION:
                conversation.messages = conversation.messages[-MAX_MESSAGES_PER_CONVERSATION:]
            conversation.updated_at = datetime.now(timezone.utc)
        else:
            conversation = Conversation(session_id=session_id, messages=[entry])
            self.session.add(conversation)

        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def get_history(self, session_id: str, limit: int = 10) -> List[dict]:
        conversation = self.get_by_session(session_id)
        if not conversation:
            return []
        return conversation.messages[-limit:]
