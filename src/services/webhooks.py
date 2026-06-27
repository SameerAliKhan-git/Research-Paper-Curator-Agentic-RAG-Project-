import hashlib
import hmac
import json
import logging
from typing import Any

import httpx
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

WEBHOOK_KEY_PREFIX = "webhooks:"
VALID_EVENTS = {"paper.ingested", "paper.synced", "ingestion.failed"}


class WebhookService:
    """Service for managing and dispatching webhook notifications."""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def register_webhook(self, url: str, events: list[str], secret: str) -> dict[str, Any]:
        """Register a new webhook endpoint."""
        invalid = set(events) - VALID_EVENTS
        if invalid:
            raise ValueError(f"Invalid events: {invalid}")

        webhook_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        data = {"url": url, "events": events, "secret": secret, "webhook_id": webhook_id}
        await self.redis.hset(f"{WEBHOOK_KEY_PREFIX}{webhook_id}", mapping={"data": json.dumps(data)})
        await self.redis.sadd(f"{WEBHOOK_KEY_PREFIX}ids", webhook_id)
        logger.info(f"Registered webhook {webhook_id} for events: {events}")
        return data

    async def notify(self, event: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Send notification to all webhooks subscribed to the given event."""
        if event not in VALID_EVENTS:
            raise ValueError(f"Invalid event: {event}")

        results = []
        webhook_ids = await self.redis.smembers(f"{WEBHOOK_KEY_PREFIX}ids")
        client = await self._get_client()

        for webhook_id in webhook_ids:
            raw = await self.redis.hget(f"{WEBHOOK_KEY_PREFIX}{webhook_id}", "data")
            if not raw:
                continue
            webhook = json.loads(raw)
            if event not in webhook.get("events", []):
                continue

            signature = self._sign_payload(json.dumps(payload), webhook["secret"])
            headers = {"X-Webhook-Event": event, "X-Webhook-Signature": signature, "Content-Type": "application/json"}

            try:
                resp = await client.post(webhook["url"], json=payload, headers=headers)
                success = resp.status_code < 300
                results.append({"webhook_id": webhook_id, "success": success, "status_code": resp.status_code})
                if not success:
                    logger.warning(f"Webhook {webhook_id} returned {resp.status_code}")
            except Exception as exc:
                results.append({"webhook_id": webhook_id, "success": False, "error": str(exc)})
                logger.error(f"Webhook {webhook_id} delivery failed: {exc}")

        return results

    def verify_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify HMAC-SHA256 signature of a webhook payload."""
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def _sign_payload(payload: str, secret: str) -> str:
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
