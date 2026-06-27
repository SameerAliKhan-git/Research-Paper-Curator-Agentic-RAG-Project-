from fastapi import Depends, HTTPException, status
from src.models.user import User
from src.services.auth.jwt_service import get_current_user
from src.config import get_settings


class RoleChecker:
    """FastAPI Dependency for Role-Based Access Control."""

    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)) -> User:
        settings = get_settings()
        
        # If authentication is globally disabled, allow all requests
        if not settings.enable_jwt_auth:
            return user

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
            )

        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: role '{user.role}' not authorized for this resource.",
            )

        return user


# Common role checks
require_admin = RoleChecker(["admin"])
require_researcher = RoleChecker(["admin", "researcher"])
require_viewer = RoleChecker(["admin", "researcher", "viewer"])
