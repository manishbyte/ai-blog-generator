"""Optional image planning and NVIDIA FLUX image generation for BlogForge.

The planner returns only a compact JSON image plan. Existing blog Markdown
is preserved and image placeholders are inserted locally in Python.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from llm.client import llm
from workflow.state import State

load_dotenv()

logger = logging.getLogger("blogforge")

MAX_IMAGES = 3
MAX_IMAGE_PLAN_TOKENS = 1400

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"

NVIDIA_IMAGE_MODEL = os.getenv(
    "NVIDIA_IMAGE_MODEL",
    "black-forest-labs/flux-3-image-klein",
)
NVIDIA_IMAGE_URL = (
    "https://ai.api.nvidia.com/v1/genai/"
    f"{NVIDIA_IMAGE_MODEL}"
)

IMAGE_STEPS = 50
IMAGE_TIMEOUT = 300
MAX_CONCURRENT_IMAGE_REQUESTS = 2


# ============================================================
# Compact image-plan schema
# ============================================================

class ImagePlanItem(BaseModel):
    """Description and placement for one optional blog image."""

    section_title: str = Field(
        description="An exact existing Markdown section heading."
    )
    placeholder: str = Field(
        description="Unique placeholder such as [[IMAGE_1]]."
    )
    filename: str = Field(
        description="Safe PNG filename, without a directory path."
    )
    alt: str = Field(description="Concise accessible image description.")
    caption: str = Field(description="Short caption for the image.")
    prompt: str = Field(description="Detailed visual-generation prompt.")
    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024",
    ] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class ImagePlanResponse(BaseModel):
    """Small response; excludes the complete blog Markdown."""

    images: list[ImagePlanItem] = Field(default_factory=list)


# ============================================================
# Image planning
# ============================================================

IMAGE_PLANNER_SYSTEM = """
You are a visual-content planner for a technical blog.

Choose zero to three images that materially improve understanding.

Prefer:
- Technical architecture diagrams
- Process and workflow diagrams
- Useful visual comparisons

Avoid:
- Decorative images with little informational value
- Repeating information already clear from the text
- Inventing facts, statistics, or unsupported claims

Return ONLY a valid JSON object in this format:
{
  "images": [
    {
      "section_title": "An exact heading supplied in the input",
      "placeholder": "[[IMAGE_1]]",
      "filename": "descriptive_name.png",
      "alt": "Accessible image description",
      "caption": "Short caption",
      "prompt": "Detailed image-generation instructions",
      "size": "1024x1024",
      "quality": "medium"
    }
  ]
}

Rules:
- The images array may be empty.
- Return no more than three images.
- Use only section headings supplied in the input.
- Keep each prompt concise but visually specific.
- Use unique placeholders and filenames.
- Use valid JSON with double-quoted keys and string values.
- Do not return Markdown fences or text outside the JSON object.
- Do not include the blog content in the response.
"""


def _extract_json(content: Any) -> dict[str, Any]:
    """Parse a JSON object returned by the model."""
    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))

        content = "".join(parts)

    text = str(content or "").strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    if not text:
        raise ValueError("The image planner returned an empty response.")

    parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError("The image plan must be a JSON object.")

    return parsed


def _get_headings(markdown: str) -> list[str]:
    """Return existing level-two and level-three Markdown headings."""
    headings = re.findall(
        r"^#{2,3}\s+(.+?)\s*#*\s*$",
        markdown,
        flags=re.MULTILINE,
    )

    return list(dict.fromkeys(
        heading.strip() for heading in headings
    ))


def _safe_slug(title: str) -> str:
    """Convert a title into a safe filename stem."""
    slug = re.sub(
        r"[^a-zA-Z0-9]+",
        "-",
        title.lower(),
    ).strip("-")

    return slug[:90] or "blogforge-blog"


def _safe_filename(filename: str) -> str:
    """Reject directory traversal and normalize image filenames."""
    filename = Path(filename).name
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)

    if not filename or filename in {".", ".."}:
        raise ValueError("Invalid image filename.")

    if not filename.lower().endswith(".png"):
        filename += ".png"

    return filename


def _insert_placeholder(
    markdown: str,
    section_title: str,
    placeholder: str,
) -> str:
    """Insert a placeholder after the first paragraph in its section."""
    lines = markdown.splitlines()

    target = section_title.strip().casefold()
    heading_index = None

    for index, line in enumerate(lines):
        match = re.match(
            r"^#{2,3}\s+(.+?)\s*#*\s*$",
            line,
        )

        if (
            match
            and match.group(1).strip().casefold() == target
        ):
            heading_index = index
            break

    if heading_index is None:
        logger.warning(
            "Image placement heading not found: %s",
            section_title,
        )
        return markdown

    content_index = heading_index + 1

    while (
        content_index < len(lines)
        and not lines[content_index].strip()
    ):
        content_index += 1

    if content_index >= len(lines):
        lines.extend(["", placeholder])
        return "\n".join(lines).strip() + "\n"

    insert_index = content_index + 1

    while insert_index < len(lines) and lines[insert_index].strip():
        if re.match(r"^#{1,6}\s+", lines[insert_index]):
            break

        insert_index += 1

    lines.insert(insert_index, "")
    lines.insert(insert_index + 1, placeholder)

    return "\n".join(lines).strip() + "\n"


async def decide_images(state: State) -> dict:
    """Plan optional images and insert placeholders into existing Markdown."""
    markdown = state.get("merged_md", "").strip()

    if not markdown:
        logger.warning("No merged Markdown available for image planning.")

        return {
            "md_with_placeholders": "",
            "image_specs": [],
        }

    headings = _get_headings(markdown)

    if not headings:
        logger.info("No section headings found; skipping image planning.")

        return {
            "md_with_placeholders": markdown + "\n",
            "image_specs": [],
        }

    plan = state.get("plan")

    planner_input = {
        "blog_title": (
            plan.blog_title
            if plan is not None
            else state.get("topic", "Technical blog")
        ),
        "topic": state.get("topic", ""),
        "section_headings": headings,
        "image_limit": MAX_IMAGES,
    }

    try:
        json_llm = llm.bind(
            response_format={"type": "json_object"},
            max_tokens=MAX_IMAGE_PLAN_TOKENS,
        )

        response = await json_llm.ainvoke(
            [
                SystemMessage(content=IMAGE_PLANNER_SYSTEM),
                HumanMessage(
                    content=(
                        "Plan useful images for this blog using only "
                        "these existing headings.\n"
                        "Return a compact JSON object with an images array.\n\n"
                        f"{json.dumps(planner_input, ensure_ascii=False)}"
                    )
                ),
            ]
        )

        raw = _extract_json(response.content)
        image_plan = ImagePlanResponse.model_validate(raw)

        selected_images = image_plan.images[:MAX_IMAGES]

        used_placeholders: set[str] = set()
        used_filenames: set[str] = set()
        valid_images: list[ImagePlanItem] = []

        for index, item in enumerate(selected_images, start=1):
            matching_heading = next(
                (
                    heading
                    for heading in headings
                    if heading.casefold()
                    == item.section_title.strip().casefold()
                ),
                None,
            )

            if matching_heading is None:
                logger.warning(
                    "Skipping image with unknown heading: %s",
                    item.section_title,
                )
                continue

            placeholder = f"[[IMAGE_{index}]]"
            filename = _safe_filename(item.filename)

            if filename.casefold() in used_filenames:
                filename = (
                    f"{Path(filename).stem}_{index}.png"
                )

            if placeholder in used_placeholders:
                continue

            normalized = item.model_copy(
                update={
                    "section_title": matching_heading,
                    "placeholder": placeholder,
                    "filename": filename,
                }
            )

            valid_images.append(normalized)
            used_placeholders.add(placeholder)
            used_filenames.add(filename.casefold())

        markdown_with_placeholders = markdown

        for item in valid_images:
            markdown_with_placeholders = _insert_placeholder(
                markdown_with_placeholders,
                item.section_title,
                item.placeholder,
            )

        logger.info(
            "Image planner selected %d image(s).",
            len(valid_images),
        )

        return {
            "md_with_placeholders": markdown_with_placeholders,
            "image_specs": [
                item.model_dump() for item in valid_images
            ],
        }

    except Exception:
        logger.exception(
            "Image planning failed; continuing without generated images."
        )

        return {
            "md_with_placeholders": markdown + "\n",
            "image_specs": [],
        }


# ============================================================
# NVIDIA FLUX.3 Image Klein generation
# ============================================================

def _nvidia_generate_image_bytes(
    prompt: str,
    size: str = "1024x1024",
) -> bytes:
    """Generate one image using NVIDIA's FLUX.3 Image Klein endpoint."""
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing; configure it in your .env file."
        )

    width, height = {
        "1024x1024": (1024, 1024),
        "1024x1536": (1024, 1536),
        "1536x1024": (1536, 1024),
    }.get(size, (1024, 1024))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": 0,
        "steps": IMAGE_STEPS,
    }

    response = requests.post(
        NVIDIA_IMAGE_URL,
        headers=headers,
        json=payload,
        timeout=IMAGE_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"NVIDIA image API failed with HTTP "
            f"{response.status_code}: {response.text}"
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "NVIDIA returned a response that was not valid JSON."
        ) from exc

    artifacts = result.get("artifacts", [])

    if not artifacts:
        raise RuntimeError(
            "NVIDIA returned no image artifacts. "
            f"Response keys: {list(result.keys())}"
        )

    image_data = artifacts[0].get("base64")

    if not image_data:
        raise RuntimeError(
            "NVIDIA's response contained no base64 image data."
        )

    try:
        if isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            image_bytes = base64.b64decode(
                image_data,
                validate=True,
            )
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "Could not decode the image returned by NVIDIA."
        ) from exc

    if not image_bytes:
        raise RuntimeError("NVIDIA returned empty image data.")

    return image_bytes


async def _generate_one_image(
    item: dict[str, Any],
    images_dir: Path,
) -> tuple[str, Path | None]:
    """Generate and save one image without blocking the event loop."""
    placeholder = str(item["placeholder"])

    try:
        filename = _safe_filename(str(item["filename"]))
        image_path = images_dir / filename

        # Reuse an existing non-empty image.
        if image_path.exists() and image_path.stat().st_size > 0:
            logger.info("Reusing existing image: %s", image_path)
            return placeholder, image_path

        prompt = str(item.get("prompt", "")).strip()

        if not prompt:
            raise ValueError("Image prompt cannot be empty.")

        size = str(item.get("size", "1024x1024"))

        image_bytes = await asyncio.to_thread(
            _nvidia_generate_image_bytes,
            prompt,
            size,
        )

        # Save generated bytes without blocking the event loop.
        await asyncio.to_thread(
            image_path.write_bytes,
            image_bytes,
        )

        logger.info("Generated NVIDIA image: %s", image_path)

        return placeholder, image_path

    except Exception:
        logger.exception(
            "NVIDIA image generation failed for %s.",
            placeholder,
        )

        return placeholder, None


# ============================================================
# Final Markdown assembly
# ============================================================

async def generate_and_place_images(state: State) -> dict:
    """Generate planned images, replace placeholders, and save the blog."""
    markdown = (
        state.get("md_with_placeholders")
        or state.get("merged_md")
        or ""
    ).strip()

    image_specs = state.get("image_specs", [])[:MAX_IMAGES]

    if not markdown:
        logger.warning("No Markdown available for final assembly.")
        return {"final": ""}

    if image_specs:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(
            MAX_CONCURRENT_IMAGE_REQUESTS
        )

        async def limited_generation(
            item: dict[str, Any],
        ) -> tuple[str, Path | None]:
            async with semaphore:
                return await _generate_one_image(
                    item,
                    IMAGES_DIR,
                )

        # Collect all image-generation results together.
        results = await asyncio.gather(
            *(
                limited_generation(item)
                for item in image_specs
            ),
            return_exceptions=True,
        )

        generated: dict[str, Path | None] = {}

        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    "Unexpected image task failure: %s",
                    result,
                )
                continue

            placeholder, path = result
            generated[placeholder] = path

        for item in image_specs:
            placeholder = str(item["placeholder"])
            image_path = generated.get(placeholder)

            if image_path is not None:
                replacement = (
                    f"![{item['alt']}](images/{image_path.name})\n\n"
                    f"*{item['caption']}*"
                )
            else:
                # Keep the blog readable when generation fails.
                replacement = (
                    f"**Image:** {item['alt']}\n\n"
                    f"*{item['caption']}*"
                )

            markdown = markdown.replace(
                placeholder,
                replacement,
            )

    # Remove any unresolved placeholders.
    markdown = re.sub(
        r"\[\[IMAGE_\d+\]\]",
        "",
        markdown,
    )
    markdown = re.sub(
        r"\n{3,}",
        "\n\n",
        markdown,
    ).strip() + "\n"

    plan = state.get("plan")

    title = (
        plan.blog_title
        if plan is not None
        else state.get("topic", "blogforge-blog")
    )

    output_path = PROJECT_ROOT / f"{_safe_slug(title)}.md"

    try:
        await asyncio.to_thread(
            output_path.write_text,
            markdown,
            encoding="utf-8",
        )

        logger.info("Saved final Markdown: %s", output_path)

    except OSError:
        logger.exception(
            "Could not save the Markdown copy at %s.",
            output_path,
        )

    return {"final": markdown}
