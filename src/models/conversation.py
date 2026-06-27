from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from src.db.interfaces.postgresql import Base

uuid4 = __import__("uuid").uuid4
utcnow = lambda: datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(String, nullable=False, index=True)
    messages = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
