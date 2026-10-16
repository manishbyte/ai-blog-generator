"""Registration, login, and current-user endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status ,Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import CurrentUser
from auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from auth.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from database.connection import get_db
from database.models import User

from datetime import datetime, timezone

from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.security import decode_access_token
from database.models import RevokedToken


router = APIRouter(tags=["Authentication"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    db: DbSession,
) -> User:
    result = await db.execute(
        select(User).where(User.email == payload.email)
    )

    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=str(payload.email).lower(),
        password_hash=hash_password(payload.password),
        is_active=True,
    )

    db.add(user)

    try:
        await db.commit()
        await db.refresh(user)
    except Exception:
        await db.rollback()
        raise

    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    db: DbSession,
) -> TokenResponse:
    email = str(payload.email).lower()

    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()

    if (
        user is None
        or not verify_password(payload.password, user.password_hash)
        or not user.is_active
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=create_access_token(user.id)
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: CurrentUser) -> User:
    return user

@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def logout(
    user: CurrentUser,
    credentials: Annotated[
        HTTPAuthorizationCredentials,
        Depends(HTTPBearer()),
    ],
    db: DbSession,
) -> Response:
    """Immediately revoke the current access token."""

    # CurrentUser has already validated this token and checked
    # that it has not previously been revoked.
    payload = decode_access_token(credentials.credentials)

    expires_at = datetime.fromtimestamp(
        payload["exp"],
        tz=timezone.utc,
    )

    db.add(
        RevokedToken(
            jti=payload["jti"],
            expires_at=expires_at,
        )
    )

    await db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)