import json
import logging
from typing import Dict, List, Optional
import redis.asyncio as aioredis

from src.config import Settings
from src.repositories.conversation import ConversationRepository
from src.database import get_db_session

logger = logging.getLogger(__name__)

# Fallback in-memory store for environments without active Redis/Postgres
_IN_MEMORY_HISTORY: Dict[str, List[dict]] = {}

class ConversationMemoryService:
    """Multi-turn conversation memory service with Redis storage and PostgreSQL/In-Memory fallback."""

    def __init__(self, settings: Settings, redis_client: Optional[aioredis.Redis] = None):
        self.settings = settings
        self.redis = redis_client
        self.redis_key_prefix = "session_chat:"
        self.ttl = 86400  # 1 day TTL for session conversations in Redis

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the conversation history for a given session."""
        entry = {"role": role, "content": content}

        # 1. Attempt Redis storage
        if self.redis:
            try:
                key = f"{self.redis_key_prefix}{session_id}"
                await self.redis.rpush(key, json.dumps(entry))
                await self.redis.expire(key, self.ttl)
                logger.debug(f"Saved message to Redis for session {session_id}")
                return
            except Exception as e:
                logger.warning(f"Redis add_message failed, falling back: {e}")

        # 2. Attempt PostgreSQL fallback
        try:
            with get_db_session() as session:
                repo = ConversationRepository(session)
                repo.create_or_append(session_id, role, content)
                logger.debug(f"Saved message to PostgreSQL for session {session_id}")
                return
        except Exception as e:
            logger.warning(f"PostgreSQL add_message failed, falling back to local memory: {e}")

        # 3. Local in-memory fallback
        if session_id not in _IN_MEMORY_HISTORY:
            _IN_MEMORY_HISTORY[session_id] = []
        _IN_MEMORY_HISTORY[session_id].append(entry)
        # Prevent unbounded growth
        if len(_IN_MEMORY_HISTORY[session_id]) > 100:
            _IN_MEMORY_HISTORY[session_id] = _IN_MEMORY_HISTORY[session_id][-100:]

    async def get_history(self, session_id: str, limit: int = 10) -> List[dict]:
        """Retrieve conversation history for a given session."""
        # 1. Attempt Redis retrieval
        if self.redis:
            try:
                key = f"{self.redis_key_prefix}{session_id}"
                raw_messages = await self.redis.lrange(key, -limit, -1)
                if raw_messages:
                    logger.debug(f"Retrieved history from Redis for session {session_id}")
                    return [json.loads(msg) for msg in raw_messages]
            except Exception as e:
                logger.warning(f"Redis get_history failed, falling back: {e}")

        # 2. Attempt PostgreSQL fallback
        try:
            with get_db_session() as session:
                repo = ConversationRepository(session)
                history = repo.get_history(session_id, limit=limit)
                if history:
                    logger.debug(f"Retrieved history from PostgreSQL for session {session_id}")
                    return [{"role": msg["role"], "content": msg["content"]} for msg in history]
        except Exception as e:
            logger.warning(f"PostgreSQL get_history failed, falling back to local memory: {e}")

        # 3. Local in-memory fallback
        return _IN_MEMORY_HISTORY.get(session_id, [])[-limit:]

    async def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a given session."""
        if self.redis:
            try:
                key = f"{self.redis_key_prefix}{session_id}"
                await self.redis.delete(key)
                logger.debug(f"Cleared Redis history for session {session_id}")
            except Exception as e:
                logger.warning(f"Redis clear_history failed: {e}")

        try:
            with get_db_session() as session:
                repo = ConversationRepository(session)
                conversation = repo.get_by_session(session_id)
                if conversation:
                    conversation.messages = []
                    session.commit()
                logger.debug(f"Cleared PostgreSQL history for session {session_id}")
        except Exception as e:
            logger.warning(f"PostgreSQL clear_history failed: {e}")

        if session_id in _IN_MEMORY_HISTORY:
            _IN_MEMORY_HISTORY[session_id] = []
