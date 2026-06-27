import argparse
import json
import logging
import os
import sys
import zipfile
from datetime import datetime
from typing import Dict, List, Any

from sqlalchemy import select, delete, text
from sqlalchemy.orm import Session

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database import init_database
from src.services.opensearch.factory import make_opensearch_client
from src.models.user import User
from src.models.paper import Paper
from src.models.conversation import Conversation
from src.models.researcher import DailyBriefing, ResearcherInterest
from src.models.collection import Collection, collection_papers
from src.models.annotation import Annotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backup_restore")


def dump_postgres_data(session: Session) -> Dict[str, List[Dict[str, Any]]]:
    """Export all PostgreSQL database tables as JSON-compatible structures."""
    logger.info("Exporting PostgreSQL tables...")
    data = {}

    # Users
    users = session.scalars(select(User)).all()
    data["users"] = [
        {
            "id": str(u.id),
            "email": u.email,
            "hashed_password": u.hashed_password,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]

    # Papers
    papers = session.scalars(select(Paper)).all()
    data["papers"] = [
        {
            "id": str(p.id),
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "pdf_url": p.pdf_url,
            "published_date": p.published_date.isoformat() if p.published_date else None,
            "categories": p.categories,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            "content_hash": p.content_hash,
            "full_text": p.full_text,
        }
        for p in papers
    ]

    # Collections
    collections = session.scalars(select(Collection)).all()
    data["collections"] = [
        {
            "id": str(c.id),
            "name": c.name,
            "description": c.description,
            "user_id": str(c.user_id),
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in collections
    ]

    # Collection Papers Many-to-Many
    stmt = select(collection_papers)
    rows = session.execute(stmt).all()
    data["collection_papers"] = [
        {
            "collection_id": str(r.collection_id),
            "paper_id": str(r.paper_id),
            "added_at": r.added_at.isoformat() if r.added_at else None,
        }
        for r in rows
    ]

    # Annotations
    annotations = session.scalars(select(Annotation)).all()
    data["annotations"] = [
        {
            "id": str(a.id),
            "paper_id": str(a.paper_id),
            "user_id": str(a.user_id),
            "text_selection": a.text_selection,
            "note": a.note,
            "tag": a.tag,
            "page": a.page,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in annotations
    ]

    # Conversations
    conversations = session.scalars(select(Conversation)).all()
    data["conversations"] = [
        {
            "id": str(c.id),
            "session_id": c.session_id,
            "messages": c.messages,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in conversations
    ]

    # Daily briefings
    briefings = session.scalars(select(DailyBriefing)).all()
    data["daily_briefings"] = [
        {
            "id": str(b.id),
            "arxiv_id": b.arxiv_id,
            "title": b.title,
            "summary": b.summary,
            "score": b.score,
            "published_date": b.published_date.isoformat(),
            "created_at": b.created_at.isoformat() if b.created_at else None,
        }
        for b in briefings
    ]

    # Researcher interests
    interests = session.scalars(select(ResearcherInterest)).all()
    data["researcher_interests"] = [
        {
            "id": str(i.id),
            "keyword": i.keyword,
            "created_at": i.created_at.isoformat() if i.created_at else None,
        }
        for i in interests
    ]

    logger.info(f"PostgreSQL tables exported. Summary: { {k: len(v) for k, v in data.items()} }")
    return data


def dump_opensearch_data() -> List[Dict[str, Any]]:
    """Export all documents from OpenSearch indexes."""
    logger.info("Exporting OpenSearch documents...")
    opensearch = make_opensearch_client()
    if not opensearch.health_check():
        logger.warning("OpenSearch service offline. Skipping index dump.")
        return []

    # Retrieve all papers/chunks using match_all scan
    query = {"query": {"match_all": {}}, "size": 10000}
    try:
        res = opensearch.client.search(index=opensearch.index_name, body=query)
        hits = res.get("hits", {}).get("hits", [])
        docs = []
        for hit in hits:
            docs.append({
                "_id": hit["_id"],
                "_source": hit["_source"]
            })
        logger.info(f"OpenSearch export complete. Ingested {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.error(f"Failed to dump OpenSearch index: {e}")
        return []


def run_backup(output_zip: str):
    """Run full system backup and package into a compressed zip file."""
    db = init_database()
    with db.get_session() as session:
        pg_data = dump_postgres_data(session)

    os_data = dump_opensearch_data()

    logger.info(f"Creating backup package: {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("postgres_backup.json", json.dumps(pg_data, indent=2))
        z.writestr("opensearch_backup.json", json.dumps(os_data, indent=2))

    logger.info("Backup process complete!")


def restore_postgres_data(session: Session, data: Dict[str, List[Dict[str, Any]]]):
    """Restore PostgreSQL tables from backup dictionary."""
    logger.info("Restoring PostgreSQL tables...")

    # Order of deletion to respect foreign key constraints
    session.execute(delete(Annotation))
    session.execute(delete(collection_papers))
    session.execute(delete(Collection))
    session.execute(delete(Paper))
    session.execute(delete(Conversation))
    session.execute(delete(DailyBriefing))
    session.execute(delete(ResearcherInterest))
    session.execute(delete(User))
    session.commit()

    # Users
    for u in data.get("users", []):
        user = User(
            id=uuid.UUID(u["id"]),
            email=u["email"],
            hashed_password=u["hashed_password"],
            role=u["role"],
            is_active=u["is_active"],
            created_at=datetime.fromisoformat(u["created_at"]) if u["created_at"] else None,
        )
        session.add(user)
    session.commit()

    # Papers
    for p in data.get("papers", []):
        paper = Paper(
            id=uuid.UUID(p["id"]),
            arxiv_id=p["arxiv_id"],
            title=p["title"],
            authors=p["authors"],
            abstract=p["abstract"],
            pdf_url=p["pdf_url"],
            published_date=datetime.fromisoformat(p["published_date"]) if p["published_date"] else None,
            categories=p["categories"],
            created_at=datetime.fromisoformat(p["created_at"]) if p["created_at"] else None,
            updated_at=datetime.fromisoformat(p["updated_at"]) if p["updated_at"] else None,
            content_hash=p["content_hash"],
            full_text=p["full_text"],
        )
        session.add(paper)
    session.commit()

    # Collections
    for c in data.get("collections", []):
        col = Collection(
            id=uuid.UUID(c["id"]),
            name=c["name"],
            description=c["description"],
            user_id=uuid.UUID(c["user_id"]),
            created_at=datetime.fromisoformat(c["created_at"]) if c["created_at"] else None,
        )
        session.add(col)
    session.commit()

    # Collection Papers
    for cp in data.get("collection_papers", []):
        stmt = collection_papers.insert().values(
            collection_id=uuid.UUID(cp["collection_id"]),
            paper_id=uuid.UUID(cp["paper_id"]),
            added_at=datetime.fromisoformat(cp["added_at"]) if cp["added_at"] else None,
        )
        session.execute(stmt)
    session.commit()

    # Annotations
    for a in data.get("annotations", []):
        ann = Annotation(
            id=uuid.UUID(a["id"]),
            paper_id=uuid.UUID(a["paper_id"]),
            user_id=uuid.UUID(a["user_id"]),
            text_selection=a["text_selection"],
            note=a["note"],
            tag=a["tag"],
            page=a["page"],
            created_at=datetime.fromisoformat(a["created_at"]) if a["created_at"] else None,
        )
        session.add(ann)
    session.commit()

    # Conversations
    for c in data.get("conversations", []):
        conv = Conversation(
            id=uuid.UUID(c["id"]),
            session_id=c["session_id"],
            messages=c["messages"],
            created_at=datetime.fromisoformat(c["created_at"]) if c["created_at"] else None,
            updated_at=datetime.fromisoformat(c["updated_at"]) if c["updated_at"] else None,
        )
        session.add(conv)
    session.commit()

    # Daily Briefings
    for b in data.get("daily_briefings", []):
        brief = DailyBriefing(
            id=uuid.UUID(b["id"]),
            arxiv_id=b["arxiv_id"],
            title=b["title"],
            summary=b["summary"],
            score=b["score"],
            published_date=datetime.fromisoformat(b["published_date"]),
            created_at=datetime.fromisoformat(b["created_at"]) if b["created_at"] else None,
        )
        session.add(brief)
    session.commit()

    # Interests
    for i in data.get("researcher_interests", []):
        interest = ResearcherInterest(
            id=uuid.UUID(i["id"]),
            keyword=i["keyword"],
            created_at=datetime.fromisoformat(i["created_at"]) if i["created_at"] else None,
        )
        session.add(interest)
    session.commit()

    logger.info("PostgreSQL restoration complete.")


def restore_opensearch_data(data: List[Dict[str, Any]]):
    """Restore OpenSearch papers/chunks index."""
    logger.info("Restoring OpenSearch index...")
    opensearch = make_opensearch_client()
    if not opensearch.health_check():
        logger.warning("OpenSearch service offline. Skipping restoration.")
        return

    # Delete index and recreate
    if opensearch.client.indices.exists(index=opensearch.index_name):
        opensearch.client.indices.delete(index=opensearch.index_name)
    opensearch.create_index()

    for doc in data:
        opensearch.client.index(
            index=opensearch.index_name,
            id=doc["_id"],
            body=doc["_source"]
        )

    logger.info(f"OpenSearch restoration complete. Restored {len(data)} documents.")


def run_restore(input_zip: str):
    """Extract backup package zip and restore PostgreSQL and OpenSearch database systems."""
    if not os.path.exists(input_zip):
        logger.error(f"Backup file not found: {input_zip}")
        sys.exit(1)

    logger.info(f"Opening backup package: {input_zip}...")
    with zipfile.ZipFile(input_zip, "r") as z:
        pg_data = json.loads(z.read("postgres_backup.json"))
        os_data = json.loads(z.read("opensearch_backup.json"))

    db = init_database()
    with db.get_session() as session:
        restore_postgres_data(session, pg_data)

    restore_opensearch_data(os_data)
    logger.info("Full system restoration complete!")


def main():
    parser = argparse.ArgumentParser(description="arXiv Paper Curator Backup and Restore Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Backup parser
    backup_parser = subparsers.add_parser("backup", help="Create a zip archive backup of Postgres and OpenSearch")
    backup_parser.add_argument("--output", default="backup.zip", help="Path to write the backup zip file")

    # Restore parser
    restore_parser = subparsers.add_parser("restore", help="Restore Postgres and OpenSearch tables from a zip archive")
    restore_parser.add_argument("--input", default="backup.zip", help="Path to backup zip archive file")

    args = parser.parse_args()

    if args.command == "backup":
        run_backup(args.output)
    elif args.command == "restore":
        run_restore(args.input)


if __name__ == "__main__":
    main()
