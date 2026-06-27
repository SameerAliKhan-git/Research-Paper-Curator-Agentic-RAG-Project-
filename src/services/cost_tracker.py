"""LLM cost tracking module for monitoring and aggregating API spend."""

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

COST_RECORDS_KEY = "cost_tracker:records"
COST_RECORDS_MAX = 10000


MODEL_PRICING: dict[str, dict[str, float]] = {
    "llama3.2": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "llama3.1": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "llama3": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "llama2": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "mistral": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "mixtral": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "codellama": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "gemma2": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "phi3": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "qwen2.5": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "deepseek-r1": {"input_per_1k": 0.0, "output_per_1k": 0.0},
    "openai/gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    "openai/gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    "openai/gpt-4-turbo": {"input_per_1k": 0.01, "output_per_1k": 0.03},
    "openai/gpt-3.5-turbo": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
    "openai/text-embedding-3-small": {"input_per_1k": 0.00002, "output_per_1k": 0.0},
    "openai/text-embedding-3-large": {"input_per_1k": 0.00013, "output_per_1k": 0.0},
    "anthropic/claude-3-5-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "anthropic/claude-3-5-haiku": {"input_per_1k": 0.0008, "output_per_1k": 0.004},
    "anthropic/claude-3-opus": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "anthropic/claude-3-haiku": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
    "jinaai/jina-embeddings-v3": {"input_per_1k": 0.00002, "output_per_1k": 0.0},
    "jinaai/jina-reranker-v2": {"input_per_1k": 0.00001, "output_per_1k": 0.0},
}


@dataclass
class CostRecord:
    """A single LLM call cost record."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """LLM cost tracker with optional Redis persistence.

    When a Redis client is provided, records are persisted to a Redis list
    so they survive restarts and are shared across workers.
    """

    def __init__(self, max_records: int = COST_RECORDS_MAX, redis_client=None):
        self._records: deque[CostRecord] = deque(maxlen=max_records)
        self._pricing: dict[str, dict[str, float]] = dict(MODEL_PRICING)
        self._redis = redis_client
        self._max_records = max_records

    def _serialize_record(self, record: CostRecord) -> str:
        return json.dumps(
            {
                "model": record.model,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "cost_usd": record.cost_usd,
                "timestamp": record.timestamp.isoformat(),
                "metadata": record.metadata,
            }
        )

    def _deserialize_record(self, data: str) -> CostRecord:
        d = json.loads(data)
        return CostRecord(
            model=d["model"],
            prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            cost_usd=d["cost_usd"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            metadata=d.get("metadata", {}),
        )

    def track_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CostRecord:
        pricing = self._pricing.get(model, {"input_per_1k": 0.0, "output_per_1k": 0.0})
        cost = (prompt_tokens / 1000.0) * pricing["input_per_1k"] + (completion_tokens / 1000.0) * pricing["output_per_1k"]
        record = CostRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 8),
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        self._records.append(record)

        # Persist to Redis asynchronously if available (fire-and-forget background task)
        if self._redis is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_to_redis(record))
            except RuntimeError:
                # No running event loop (e.g. running in synchronous tests)
                pass

        logger.debug(
            "Tracked LLM call: model=%s tokens=%d+%d cost=$%.6f",
            model,
            prompt_tokens,
            completion_tokens,
            record.cost_usd,
        )
        return record

    async def _persist_to_redis(self, record: CostRecord):
        """Helper to write records to Redis asynchronously."""
        try:
            await self._redis.rpush(COST_RECORDS_KEY, self._serialize_record(record))
            await self._redis.ltrim(COST_RECORDS_KEY, -self._max_records, -1)
        except Exception as e:
            logger.debug(f"Failed to persist cost record to Redis: {e}")

    async def load_from_redis(self) -> int:
        """Load persisted records from Redis into the in-memory deque.

        Returns the number of records loaded.
        """
        if self._redis is None:
            return 0
        try:
            raw_records = await self._redis.lrange(COST_RECORDS_KEY, 0, -1)
            for raw in raw_records:
                record = self._deserialize_record(raw)
                self._records.append(record)
            logger.info(f"Loaded {len(raw_records)} cost records from Redis")
            return len(raw_records)
        except Exception as e:
            logger.warning(f"Failed to load cost records from Redis: {e}")
            return 0

    def get_total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self._records), 8)

    def get_cost_by_model(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for record in self._records:
            totals[record.model] += record.cost_usd
        return {k: round(v, 8) for k, v in totals.items()}

    def get_recent_records(self, n: int = 100) -> list[CostRecord]:
        return list(self._records)[-n:]

    def get_total_tokens(self) -> dict[str, int]:
        prompt_total = sum(r.prompt_tokens for r in self._records)
        completion_total = sum(r.completion_tokens for r in self._records)
        return {"prompt_tokens": prompt_total, "completion_tokens": completion_total}


cost_tracker = CostTracker()


class CostMiddleware(BaseHTTPMiddleware):
    """Middleware that tracks per-request LLM costs and adds cost headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request.state.llm_cost_usd = 0.0
        response = await call_next(request)
        response.headers["X-LLM-Cost-USD"] = f"{request.state.llm_cost_usd:.6f}"
        return response
