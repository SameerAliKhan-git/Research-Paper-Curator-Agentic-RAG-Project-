import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID
from src.db.interfaces.postgresql import Base

utcnow = lambda: datetime.now(timezone.utc)


class ResearcherInterest(Base):
    __tablename__ = "researcher_interests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    keyword = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=utcnow)


class DailyBriefing(Base):
    __tablename__ = "daily_briefings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    arxiv_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    published_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=utcnow)
