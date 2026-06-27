import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from src.db.interfaces.postgresql import Base

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    text_selection = Column(Text, nullable=True)  # Selected text from PDF
    note = Column(Text, nullable=False)  # Researcher's annotation/note
    tag = Column(String, nullable=True, index=True)  # e.g. "methodology", "limitation", "key result"
    page = Column(Integer, nullable=True)  # Page number of PDF
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
