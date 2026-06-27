import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from src.models.annotation import Annotation

logger = logging.getLogger(__name__)


class AnnotationRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        paper_id: UUID,
        user_id: UUID,
        note: str,
        text_selection: Optional[str] = None,
        tag: Optional[str] = None,
        page: Optional[int] = None,
    ) -> Annotation:
        """Create a new annotation on a paper."""
        annotation = Annotation(
            paper_id=paper_id,
            user_id=user_id,
            note=note,
            text_selection=text_selection,
            tag=tag,
            page=page
        )
        self.session.add(annotation)
        self.session.commit()
        self.session.refresh(annotation)
        return annotation

    def get_by_id(self, annotation_id: UUID) -> Optional[Annotation]:
        """Retrieve an annotation by ID."""
        stmt = select(Annotation).where(Annotation.id == annotation_id)
        return self.session.scalar(stmt)

    def get_all_by_paper(self, paper_id: UUID) -> List[Annotation]:
        """Retrieve all annotations for a specific paper."""
        stmt = select(Annotation).where(Annotation.paper_id == paper_id).order_by(Annotation.created_at.desc())
        return list(self.session.scalars(stmt))

    def get_all_by_user(self, user_id: UUID) -> List[Annotation]:
        """Retrieve all annotations made by a specific user."""
        stmt = select(Annotation).where(Annotation.user_id == user_id).order_by(Annotation.created_at.desc())
        return list(self.session.scalars(stmt))

    def update(self, annotation: Annotation) -> Annotation:
        """Save updates to an annotation."""
        self.session.add(annotation)
        self.session.commit()
        self.session.refresh(annotation)
        return annotation

    def delete(self, annotation_id: UUID) -> bool:
        """Delete an annotation."""
        annotation = self.get_by_id(annotation_id)
        if annotation:
            self.session.delete(annotation)
            self.session.commit()
            return True
        return False
