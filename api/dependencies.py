"""Authentication dependencies for protected endpoints."""

from datetime import datetime, timezone
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.security import decode_access_token
from database.connection import get_db
from database.models import RevokedToken, User


bearer_scheme = HTTPBearer(auto_error=False)


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid, expired, or revoked access token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    if credentials is None:
        raise unauthorized()

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = UUID(payload["sub"])
        token_id = payload["jti"]
    except Exception:
        raise unauthorized() from None

    revoked_result = await db.execute(
        select(RevokedToken.id).where(RevokedToken.jti == token_id)
    )
    if revoked_result.scalar_one_or_none() is not None:
        raise unauthorized()

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise unauthorized()

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]