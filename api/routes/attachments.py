"""Secure retrieval of stored message attachments."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import CurrentUser
from database.connection import get_db
from database.models import Conversation, Message, MessageAttachment


router = APIRouter(tags=["Attachments"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.get("/{attachment_id}")
async def get_attachment(
    attachment_id: UUID,
    user: CurrentUser,
    db: DbSession,
) -> Response:
    result = await db.execute(
        select(MessageAttachment)
        .join(
            Message,
            MessageAttachment.message_id == Message.id,
        )
        .join(
            Conversation,
            Message.conversation_id == Conversation.id,
        )
        .where(
            MessageAttachment.id == attachment_id,
            Conversation.user_id == user.id,
        )
    )

    attachment = result.scalar_one_or_none()

    if attachment is None:
        raise HTTPException(
            status_code=404,
            detail="Attachment not found.",
        )

    return Response(
        content=attachment.image_data,
        media_type=attachment.mime_type,
        headers={
            "Content-Disposition": (
                f'inline; filename="{attachment.filename}"'
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )