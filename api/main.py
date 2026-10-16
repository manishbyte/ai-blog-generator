"""BlogForge FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.attachments import router as attachments_router
from api.routes.auth import router as auth_router
from api.routes.conversations import router as conversations_router
from database.init_db import close_db, init_db
from api.routes.generated_images import router as generated_images_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    try:
        yield
    finally:
        await close_db()


app = FastAPI(
    title="BlogForge API",
    description="Authentication, conversations, and blog generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router, prefix="/auth")
app.include_router(conversations_router, prefix="/conversations")
app.include_router(attachments_router, prefix="/attachments")
app.include_router(
    generated_images_router,
    prefix="/generated-images",
)


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}