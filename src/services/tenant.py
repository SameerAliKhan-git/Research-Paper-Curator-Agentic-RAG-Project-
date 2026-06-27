import logging
import re
from typing import Optional

from fastapi import Depends, HTTPException, Request
from src.services.auth.api_key_service import APIKeyMetadata, require_api_key
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Restrict tenant IDs to safe characters
_TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class TenantContext:
    """Holds tenant identity and generates tenant-scoped filters."""

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id

    def get_tenant_filter(self) -> dict:
        """Return an OpenSearch term filter for the tenant."""
        return {"term": {"tenant_id": self.tenant_id}}

    def get_tenant_prefix(self) -> str:
        """Return a Redis key prefix for the tenant."""
        return f"tenant:{self.tenant_id}:"


async def require_tenant(
    request: Request,
    api_key_meta: APIKeyMetadata = Depends(require_api_key),
) -> TenantContext:
    """FastAPI dependency that enforces tenant authorization.

    Validates tenant ID format and checks it against the authenticated user's allowed tenants.
    """
    tenant_id = request.headers.get("X-Tenant-ID", "default")

    # Validate format
    if not _TENANT_ID_PATTERN.match(tenant_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid X-Tenant-ID format.",
        )

    # Check against allowed tenants for this API key
    if api_key_meta.tenants is not None:
        if tenant_id not in api_key_meta.tenants:
            logger.warning(f"Tenant {tenant_id!r} not allowed for user {api_key_meta.user_id!r}")
            raise HTTPException(
                status_code=403,
                detail=f"Tenant '{tenant_id}' is not authorized for this API key.",
            )

    tenant_ctx = TenantContext(tenant_id=tenant_id)
    request.state.tenant = tenant_ctx
    return tenant_ctx


class TenantMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that appends the X-Tenant-ID header to the response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        tenant_ctx = getattr(request.state, "tenant", None)
        if tenant_ctx is not None:
            response.headers["X-Tenant-ID"] = tenant_ctx.tenant_id
        return response
