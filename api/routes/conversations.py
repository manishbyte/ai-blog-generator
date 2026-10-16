
"""Conversation and blog-generation endpoints."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.schemas.conversations import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    GenerateMessageRequest,
    GenerateMessageResponse,
    MessageResponse,
)
from auth.dependencies import CurrentUser
from database.connection import get_db
from database.models import Conversation, Message
from guardrails.topic_injection import detect_topic_injection
from workflow.graph import generate_blog


router = APIRouter(tags=["Conversations"])

DbSession = Annotated[AsyncSession, Depends(get_db)]

logger = logging.getLogger(__name__)

BLOCKED_TOPIC_MESSAGE = (
    "This request was blocked by BlogForge security guardrails. "
    "Please submit a legitimate blog topic without instructions to "
    "override security rules or disclose confidential information."
)


async def get_owned_conversation(
    conversation_id: UUID,
    user_id: UUID,
    db: AsyncSession,
) -> Conversation:
    """Return a conversation only when it belongs to the user."""

    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )

    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return conversation


async def get_message_with_attachments(
    message_id: UUID,
    db: AsyncSession,
) -> Message:
    """Load a message and its attachments without async lazy loading."""

    result = await db.execute(
        select(Message)
        .options(selectinload(Message.attachments))
        .where(Message.id == message_id)
    )

    message = result.scalar_one_or_none()

    if message is None:
        raise RuntimeError(f"Message {message_id} was not found.")

    return message


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: ConversationCreate,
    user: CurrentUser,
    db: DbSession,
) -> Conversation:
    """Create a conversation for the authenticated user."""

    title = (payload.title or "New blog").strip() or "New blog"

    conversation = Conversation(
        user_id=user.id,
        title=title,
    )

    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    return conversation


@router.get(
    "",
    response_model=list[ConversationResponse],
)
async def list_conversations(
    user: CurrentUser,
    db: DbSession,
) -> list[Conversation]:
    """List conversations belonging to the authenticated user."""

    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )

    return list(result.scalars().all())


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
async def get_conversation(
    conversation_id: UUID,
    user: CurrentUser,
    db: DbSession,
) -> Conversation:
    """Retrieve conversation history with messages and attachments."""

    result = await db.execute(
        select(Conversation)
        .options(
            selectinload(Conversation.messages).selectinload(
                Message.attachments
            )
        )
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )

    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    conversation.messages.sort(key=lambda message: message.created_at)

    return conversation


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    user: CurrentUser,
    db: DbSession,
) -> None:
    """Delete a conversation owned by the authenticated user."""

    conversation = await get_owned_conversation(
        conversation_id,
        user.id,
        db,
    )

    await db.delete(conversation)
    await db.commit()


@router.post(
    "/{conversation_id}/messages",
    response_model=GenerateMessageResponse,
)
async def generate_message(
    conversation_id: UUID,
    payload: GenerateMessageRequest,
    user: CurrentUser,
    db: DbSession,
) -> GenerateMessageResponse:
    """Validate a topic, generate a blog, and persist the response."""

    conversation = await get_owned_conversation(
        conversation_id,
        user.id,
        db,
    )

    topic = payload.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Blog topic cannot be empty.",
        )

    # SECURITY: Check the raw user topic before saving it or invoking
    # the LangGraph workflow, research agent, workers, or image generator.
    injection_reason = detect_topic_injection(topic)

    if injection_reason:
        logger.warning(
            "Blocked suspicious blog topic for user_id=%s; categories=%s",
            user.id,
            injection_reason,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=BLOCKED_TOPIC_MESSAGE,
        )

    # Save only topics that pass the input guardrail.
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=topic,
        extra_data={},
    )

    db.add(user_message)

    if conversation.title == "New blog":
        conversation.title = topic[:200]

    await db.commit()

    try:
        # Run the existing LangGraph workflow only after the guardrail passes.
        result = await generate_blog(
            topic,
            as_of=payload.as_of,
        )

        # Prefer Markdown containing the completed image-placement stage.
        blog_content = str(
            result.get("final")
            or result.get("merged_md")
            or ""
        ).strip()

        if not blog_content:
            raise RuntimeError(
                "The workflow completed without producing final Markdown."
            )

        image_specs = result.get("image_specs") or []

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=blog_content,
            extra_data={"image_specs": image_specs},
        )

        db.add(assistant_message)
        await db.commit()

    except Exception as exc:
        await db.rollback()

        logger.exception("Blog generation or persistence failed.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Blog generation failed. Check the backend logs.",
        ) from exc

    # Load attachment relationships explicitly to avoid MissingGreenlet
    # during Pydantic response serialization.
    try:
        user_message = await get_message_with_attachments(
            user_message.id,
            db,
        )

        assistant_message = await get_message_with_attachments(
            assistant_message.id,
            db,
        )

        await db.refresh(conversation)

        return GenerateMessageResponse(
            conversation=ConversationResponse.model_validate(conversation),
            user_message=MessageResponse.model_validate(user_message),
            assistant_message=MessageResponse.model_validate(assistant_message),
        )

    except Exception:
        logger.exception(
            "Failed to construct the blog-generation response."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The blog was generated, but its response could not be loaded."
            ),
        ) from None
