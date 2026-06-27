import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from src.models.collection import Collection, collection_papers
from src.models.paper import Paper

logger = logging.getLogger(__name__)


class CollectionRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str, user_id: UUID, description: Optional[str] = None) -> Collection:
        """Create a new collection for a user."""
        collection = Collection(name=name, user_id=user_id, description=description)
        self.session.add(collection)
        self.session.commit()
        self.session.refresh(collection)
        return collection

    def get_by_id(self, collection_id: UUID) -> Optional[Collection]:
        """Retrieve a collection by ID."""
        stmt = select(Collection).where(Collection.id == collection_id)
        return self.session.scalar(stmt)

    def get_all_by_user(self, user_id: UUID) -> List[Collection]:
        """Retrieve all collections belonging to a user."""
        stmt = select(Collection).where(Collection.user_id == user_id).order_by(Collection.created_at.desc())
        return list(self.session.scalars(stmt))

    def delete(self, collection_id: UUID) -> bool:
        """Delete a collection."""
        collection = self.get_by_id(collection_id)
        if collection:
            self.session.delete(collection)
            self.session.commit()
            return True
        return False

    def add_paper_to_collection(self, collection_id: UUID, paper_id: UUID) -> bool:
        """Add a paper to a collection."""
        try:
            # Check if association already exists
            stmt = select(collection_papers).where(
                collection_papers.c.collection_id == collection_id,
                collection_papers.c.paper_id == paper_id
            )
            existing = self.session.execute(stmt).first()
            if existing:
                return True

            # Insert association
            stmt_insert = collection_papers.insert().values(collection_id=collection_id, paper_id=paper_id)
            self.session.execute(stmt_insert)
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add paper to collection: {e}")
            self.session.rollback()
            return False

    def remove_paper_from_collection(self, collection_id: UUID, paper_id: UUID) -> bool:
        """Remove a paper from a collection."""
        try:
            stmt = delete(collection_papers).where(
                collection_papers.c.collection_id == collection_id,
                collection_papers.c.paper_id == paper_id
            )
            result = self.session.execute(stmt)
            self.session.commit()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to remove paper from collection: {e}")
            self.session.rollback()
            return False

    def get_papers_in_collection(self, collection_id: UUID) -> List[Paper]:
        """Get all papers inside a collection."""
        stmt = (
            select(Paper)
            .join(collection_papers, Paper.id == collection_papers.c.paper_id)
            .where(collection_papers.c.collection_id == collection_id)
            .order_by(collection_papers.c.added_at.desc())
        )
        return list(self.session.scalars(stmt))
