
"""Command-line entry point for BlogForge."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from guardrails import InputValidationError, require_valid_topic
from workflow.graph import generate_blog


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = BASE_DIR / "images"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("blogforge")


# ============================================================
# Blog generation
# ============================================================

async def run_blog_generation(
    topic: str,
    as_of: str | None = None,
) -> Path:
    """Validate the topic, generate a blog, and save its Markdown."""

    # Phase 1 guardrail: validate and normalize user input.
    topic = require_valid_topic(topic)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting BlogForge...")
    logger.info("Topic: %s", topic)
    logger.info("Markdown output directory: %s", OUTPUT_DIR.resolve())
    logger.info("Image output directory: %s", IMAGES_DIR.resolve())

    # Run the LangGraph workflow.
    result = await generate_blog(topic, as_of=as_of)

    final_markdown = str(result.get("final") or "").strip()

    if not final_markdown:
        raise RuntimeError(
            "The workflow completed without producing final Markdown."
        )

    plan = result.get("plan")
    title = getattr(plan, "blog_title", None) or topic

    # Reuse the existing filename utility.
    from agents.image_agent import _safe_slug

    slug = _safe_slug(title)
    output_path = OUTPUT_DIR / f"{slug}.md"

    # Save the final blog Markdown.
    output_path.write_text(
        final_markdown + "\n",
        encoding="utf-8",
    )

    # Report image specifications returned by the workflow.
    image_specs = result.get("image_specs") or []

    logger.info("Blog generation completed.")
    logger.info("Saved Markdown: %s", output_path.resolve())

    if image_specs:
        logger.info(
            "Image specifications returned by workflow: %d",
            len(image_specs),
        )
        logger.info("Images directory: %s", IMAGES_DIR.resolve())
    else:
        logger.info("No images were requested for this blog.")

    # Display a short run summary.
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(final_markdown)
    print("=" * 72)
    print(f"\nMarkdown saved to: {output_path.resolve()}")
    print(f"Images directory:  {IMAGES_DIR.resolve()}")

    return output_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a blog using the BlogForge LangGraph workflow."
    )

    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Topic for the blog.",
    )

    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Reference date in YYYY-MM-DD format. Defaults to today.",
    )

    return parser.parse_args()


async def async_main() -> int:
    load_dotenv(BASE_DIR / ".env")
    args = parse_args()

    try:
        await run_blog_generation(
            topic=args.topic,
            as_of=args.as_of,
        )
        return 0

    
    except InputValidationError as exc:
        print("\nBlog topic rejected.")
        print(f"Reason: {exc}")
        print("Please correct the topic and try again.")
        return 2


    except KeyboardInterrupt:
        logger.warning("Blog generation cancelled.")
        return 130

    except Exception:
        logger.exception("Blog generation failed.")
        return 1


def main() -> None:
    try:
        exit_code = asyncio.run(async_main())
    except KeyboardInterrupt:
        exit_code = 130

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
