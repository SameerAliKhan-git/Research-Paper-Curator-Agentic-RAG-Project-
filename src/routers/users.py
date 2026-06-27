import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.dependencies import get_db_session
from src.models.user import User
from src.schemas.api.users import TokenResponse, UserCreate, UserResponse
from src.services.auth.jwt_service import create_access_token, get_current_user, hash_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(request: UserCreate, db: Session = Depends(get_db_session)) -> User:
    """Register a new user account."""
    # Check if email exists
    stmt = select(User).where(User.email == request.email)
    existing_user = db.scalar(stmt)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email is already registered",
        )

    # First user is admin, others are researchers
    stmt_count = select(User)
    total_users = len(db.scalars(stmt_count).all())
    role = "admin" if total_users == 0 else "researcher"

    hashed_pw = hash_password(request.password)
    user = User(email=request.email, hashed_password=hashed_pw, role=role)
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info(f"Registered new user: {user.email} with role: {role}")
    return user


@router.post("/login", response_model=TokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db_session)
) -> dict:
    """Login to retrieve a JWT access token (supports form submit in Swagger UI)."""
    stmt = select(User).where(User.email == form_data.username)
    user = db.scalar(stmt)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    # Generate token with user email
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)) -> User:
    """Retrieve details of currently logged-in user profile."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT Authentication is currently disabled in app settings",
        )
    return current_user
