import os
import uuid
import pytest
from sqlalchemy import select
import src.database as _db_module
from src.database import init_database
from src.models.user import User
from src.models.paper import Paper
from src.repositories.collection import CollectionRepository
from src.repositories.annotation import AnnotationRepository
from src.services.auth.jwt_service import hash_password, verify_password


def is_db_reachable() -> bool:
    """Check if Postgres is reachable, trying configured settings first, then docker-compose localhost fallback."""
    try:
        db = init_database()
        with db.get_session() as session:
            session.execute(select(1))
            return True
    except Exception:
        pass

    # Try default docker-compose localhost port-mapped credentials
    try:
        _db_module._database = None  # Reset singleton
        os.environ["POSTGRES_DATABASE_URL"] = "postgresql+psycopg2://rag_user:rag_password@localhost:5432/rag_db"
        import src.config as _config_module
        _config_module._settings_cache = None
        db = init_database()
        with db.get_session() as session:
            session.execute(select(1))
            return True
    except Exception:
        pass

    return False


@pytest.mark.skipif(not is_db_reachable(), reason="Database service not running")
def test_user_creation_and_auth():
    db = init_database()
    with db.get_session() as session:
        # 1. Clean test user if exists
        test_email = "pytest_user@example.com"
        session.execute(select(User).where(User.email == test_email))
        existing = session.scalars(select(User).where(User.email == test_email)).first()
        if existing:
            session.delete(existing)
            session.commit()

        # 2. Create user
        hashed = hash_password("secret123")
        user = User(email=test_email, hashed_password=hashed, role="researcher")
        session.add(user)
        session.commit()
        session.refresh(user)

        assert user.id is not None
        assert user.email == test_email
        assert verify_password("secret123", user.hashed_password)

        # Cleanup
        session.delete(user)
        session.commit()


@pytest.mark.skipif(not is_db_reachable(), reason="Database service not running")
def test_collections_and_annotations_crud():
    db = init_database()
    with db.get_session() as session:
        # Create temp user
        user = User(email=f"temp_{uuid.uuid4()}@example.com", hashed_password="hashed_pwd", role="researcher")
        session.add(user)
        session.commit()

        from datetime import datetime, timezone
        # Create temp paper
        paper = Paper(
            arxiv_id=f"test_{uuid.uuid4()}",
            title="Test Paper Title",
            authors=["John Doe"],
            abstract="Abstract of test paper",
            pdf_url="http://example.com/test.pdf",
            categories=["cs.AI"],
            published_date=datetime.now(timezone.utc)
        )
        session.add(paper)
        session.commit()

        # 1. Test Collection Repository
        col_repo = CollectionRepository(session)
        collection = col_repo.create(name="AI Research", user_id=user.id, description="AI models")
        
        assert collection.id is not None
        assert collection.name == "AI Research"
        assert collection.user_id == user.id

        # Add paper to collection
        success = col_repo.add_paper_to_collection(collection.id, paper.id)
        assert success is True

        # Get papers in collection
        papers_in_col = col_repo.get_papers_in_collection(collection.id)
        assert len(papers_in_col) == 1
        assert papers_in_col[0].id == paper.id

        # 2. Test Annotation Repository
        ann_repo = AnnotationRepository(session)
        annotation = ann_repo.create(
            paper_id=paper.id,
            user_id=user.id,
            note="This is a key methodology detail.",
            text_selection="We propose a new model",
            tag="methodology",
            page=2
        )

        assert annotation.id is not None
        assert annotation.note == "This is a key methodology detail."
        assert annotation.tag == "methodology"

        # List annotations for paper
        paper_annotations = ann_repo.get_all_by_paper(paper.id)
        assert len(paper_annotations) == 1
        assert paper_annotations[0].id == annotation.id

        # Delete annotation
        assert ann_repo.delete(annotation.id) is True
        assert ann_repo.get_by_id(annotation.id) is None

        # Remove paper from collection
        assert col_repo.remove_paper_from_collection(collection.id, paper.id) is True
        assert len(col_repo.get_papers_in_collection(collection.id)) == 0

        # Delete collection
        assert col_repo.delete(collection.id) is True
        assert col_repo.get_by_id(collection.id) is None

        # Clean user and paper
        session.delete(paper)
        session.delete(user)
        session.commit()
