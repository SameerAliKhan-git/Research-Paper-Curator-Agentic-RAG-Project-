from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.models.paper import Paper
from src.schemas.arxiv.paper import PaperCreate


class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, paper: PaperCreate) -> Paper:
        db_paper = Paper(**paper.model_dump())
        self.session.add(db_paper)
        self.session.commit()
        self.session.refresh(db_paper)
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)
        return self.session.scalar(stmt)

    def get_by_content_hash(self, content_hash: str) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.content_hash == content_hash)
        return self.session.scalar(stmt)

    def get_by_id(self, paper_id: UUID) -> Optional[Paper]:
        stmt = select(Paper).where(Paper.id == paper_id)
        return self.session.scalar(stmt)

    def get_all(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        stmt = select(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_count(self) -> int:
        stmt = select(func.count(Paper.id))
        return self.session.scalar(stmt) or 0

    def get_processed_count(self) -> int:
        stmt = select(func.count(Paper.id)).where(Paper.pdf_processed == True)
        return self.session.scalar(stmt) or 0

    def get_processed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that have been successfully processed with PDF content."""
        stmt = (
            select(Paper)
            .where(Paper.pdf_processed == True)
            .order_by(Paper.pdf_processing_date.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt))

    def get_unprocessed_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that haven't been processed for PDF content yet."""
        stmt = select(Paper).where(Paper.pdf_processed == False).order_by(Paper.published_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_papers_with_raw_text(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """Get papers that have raw text content stored."""
        stmt = select(Paper).where(Paper.raw_text != None).order_by(Paper.pdf_processing_date.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def get_processing_stats(self) -> dict:
        """Get statistics about PDF processing status."""
        total_papers = self.get_count()

        # Count processed papers
        processed_stmt = select(func.count(Paper.id)).where(Paper.pdf_processed == True)
        processed_papers = self.session.scalar(processed_stmt) or 0

        # Count papers with text
        text_stmt = select(func.count(Paper.id)).where(Paper.raw_text != None)
        papers_with_text = self.session.scalar(text_stmt) or 0

        return {
            "total_papers": total_papers,
            "processed_papers": processed_papers,
            "papers_with_text": papers_with_text,
            "processing_rate": (processed_papers / total_papers * 100) if total_papers > 0 else 0,
            "text_extraction_rate": (papers_with_text / processed_papers * 100) if processed_papers > 0 else 0,
        }

    def update(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def get_category_trends(self, category: str | None = None, months_back: int = 12) -> list[dict]:
        """Return monthly paper counts grouped by category."""
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import String, cast, func

        cutoff = datetime.now(timezone.utc) - timedelta(days=months_back * 30)

        category_col = Paper.categories
        month_expr = func.date_trunc("month", Paper.published_date).label("month")

        stmt = (
            select(
                month_expr,
                func.jsonb_array_elements_text(category_col).label("category"),
                func.count(Paper.id).label("count"),
            )
            .where(Paper.published_date >= cutoff)
            .group_by(month_expr, func.jsonb_array_elements_text(category_col))
            .order_by(month_expr.desc())
        )

        if category:
            stmt = stmt.having(func.jsonb_array_elements_text(category_col) == category)

        rows = self.session.execute(stmt).all()
        return [
            {
                "month": row.month.strftime("%Y-%m") if row.month else "unknown",
                "category": row.category,
                "count": row.count,
            }
            for row in rows
        ]

    def get_papers_citing(self, arxiv_id: str) -> List[Paper]:
        """Get papers whose references list contains the given arxiv_id."""
        stmt = select(Paper).where(Paper.references.isnot(None))
        papers = list(self.session.scalars(stmt))
        return [p for p in papers if isinstance(p.references, list) and arxiv_id in p.references]

    def upsert(self, paper_create: PaperCreate) -> Paper:
        # Check if paper already exists
        existing_paper = self.get_by_arxiv_id(paper_create.arxiv_id)
        if existing_paper:
            # Update existing paper with new content
            for key, value in paper_create.model_dump(exclude_unset=True).items():
                setattr(existing_paper, key, value)
            return self.update(existing_paper)
        else:
            # Create new paper
            return self.create(paper_create)
