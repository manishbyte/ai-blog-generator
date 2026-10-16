"""Serve generated BlogForge images to authenticated users."""

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from auth.dependencies import CurrentUser

router = APIRouter(tags=["Generated Images"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = (PROJECT_ROOT / "images").resolve()

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@router.get("/{filename}")
async def get_generated_image(
    filename: str,
    user: CurrentUser,
) -> FileResponse:
    """Return a generated image to an authenticated user."""

    safe_filename = Path(unquote(filename)).name

    if (
        safe_filename != filename
        or not safe_filename
        or safe_filename in {".", ".."}
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image filename.",
        )

    image_path = (IMAGES_DIR / safe_filename).resolve()

    if (
        image_path.parent != IMAGES_DIR
        or image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES
        or not image_path.is_file()
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Generated image not found.",
        )

    return FileResponse(
        path=image_path,
        media_type={
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }[image_path.suffix.lower()],
        headers={"X-Content-Type-Options": "nosniff"},
    )
