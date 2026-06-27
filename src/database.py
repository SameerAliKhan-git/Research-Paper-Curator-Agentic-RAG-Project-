import threading
from contextlib import contextmanager
from typing import Optional

from src.db.factory import make_database

# Global database instance — initialized once via init_database()
_database: Optional[object] = None
_database_lock = threading.Lock()


def init_database():
    """Initialize the global database singleton. Called once during app startup."""
    global _database
    with _database_lock:
        if _database is None:
            _database = make_database()
    return _database


def get_database():
    """Get the global database instance. Raises if not initialized."""
    if _database is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _database


@contextmanager
def get_db_session():
    """Get a database session context manager."""
    database = get_database()
    with database.get_session() as session:
        yield session
