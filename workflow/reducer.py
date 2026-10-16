
"""Reducer nodes and output guardrails for BlogForge."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from workflow.state import State
from guardrails.prompt_injection import scan_untrusted_content
from agents.image_agent import (
    decide_images,
    generate_and_place_images,
)

logger = logging.getLogger("blogforge")

MAX_OUTPUT_CHARS = 100_000

# Matches standard Markdown links, excluding Markdown images.
MARKDOWN_LINK_PATTERN = re.compile(
    r"(?<!!)\[([^\]]+)\]\((https?://[^\s)]+)(?:\s+[^)]*)?\)"
)

FENCED_CODE_PATTERN = re.compile(
    r"(```.*?```|~~~.*?~~~)",
    flags=re.DOTALL,
)


def _canonical_url(url: str) -> str | None:
    """Normalize a URL for comparison without changing its destination."""
    try:
        parts = urlsplit(url.strip())

        if parts.scheme.lower() not in {"http", "https"}:
            return None

        if not parts.hostname:
            return None

        # Compare hostnames case-insensitively and ignore URL fragments.
        hostname = parts.hostname.lower()
        port = parts.port

        if port is not None:
            default_port = (
                parts.scheme.lower() == "http" and port == 80
            ) or (
                parts.scheme.lower() == "https" and port == 443
            )
            netloc = hostname if default_port else f"{hostname}:{port}"
        else:
            netloc = hostname

        if parts.username or parts.password:
            return None

        path = parts.path.rstrip("/") or "/"

        return urlunsplit((
            parts.scheme.lower(),
            netloc,
            path,
            parts.query,
            "",
        ))

    except (TypeError, ValueError):
        return None


def _trusted_evidence_urls(state: State) -> set[str]:
    """Build the set of URLs supplied by the research stage."""
    evidence = state.get("evidence", [])

    if not isinstance(evidence, list):
        return set()

    trusted_urls: set[str] = set()

    for item in evidence:
        if hasattr(item, "url"):
            url = item.url
        elif isinstance(item, dict):
            url = item.get("url")
        else:
            continue

        if not isinstance(url, str):
            continue

        canonical = _canonical_url(url)

        if canonical:
            trusted_urls.add(canonical)

    return trusted_urls


def _validate_markdown_links(
    markdown: str,
    trusted_urls: set[str],
) -> tuple[str, int]:
    """Remove unsupported external Markdown hyperlinks.

    The visible link text is preserved. Fenced code blocks are not modified.
    Markdown images and relative links are left unchanged.
    """
    parts = FENCED_CODE_PATTERN.split(markdown)
    removed_count = 0

    for index, part in enumerate(parts):
        # Odd-indexed parts are fenced code blocks.
        if index % 2 == 1:
            continue

        def replace_link(match: re.Match[str]) -> str:
            nonlocal removed_count

            label = match.group(1)
            url = match.group(2)
            canonical = _canonical_url(url)

            if canonical and canonical in trusted_urls:
                return match.group(0)

            removed_count += 1

            logger.warning(
                "Removed unsupported external Markdown link: %s",
                url,
            )

            # Preserve the link label without presenting it as a verified link.
            return label

        parts[index] = MARKDOWN_LINK_PATTERN.sub(
            replace_link,
            part,
        )

    return "".join(parts), removed_count


def _check_expected_structure(
    markdown: str,
    state: State,
) -> list[str]:
    """Report missing title/section headings without discarding the blog."""
    issues: list[str] = []

    plan = state.get("plan")
    if plan is None:
        return ["The output has no associated plan."]

    lines = markdown.splitlines()
    first_heading = next(
        (
            line.strip()
            for line in lines
            if line.strip().startswith("# ")
        ),
        None,
    )

    expected_title = f"# {plan.blog_title}".strip()

    if first_heading != expected_title:
        issues.append(
            f"Expected blog title heading {expected_title!r}; "
            f"found {first_heading!r}."
        )

    for task in plan.tasks:
        expected_heading = f"## {task.title}".strip()

        if not any(
            line.strip() == expected_heading
            for line in lines
        ):
            issues.append(
                f"Expected section heading is missing: {expected_heading}"
            )

    return issues


def _apply_output_guardrails(
    markdown: str,
    state: State,
) -> str:
    """Validate merged Markdown before it enters the image stage."""
    if not isinstance(markdown, str):
        raise TypeError("Merged blog output must be a string.")

    markdown = markdown.strip()

    if not markdown:
        raise ValueError("Merged blog output is empty.")

    if len(markdown) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"Merged blog exceeds the maximum output size "
            f"of {MAX_OUTPUT_CHARS} characters."
        )

    # Check for suspicious patterns. This is a warning, not proof of an attack.
    scan = scan_untrusted_content(markdown)

    if scan.suspicious:
        logger.warning(
            "Potential prompt-injection patterns found in generated Markdown. "
            "Categories: %s",
            ", ".join(scan.matched_categories),
        )

    # Validate external Markdown citations/links against retrieved evidence.
    trusted_urls = _trusted_evidence_urls(state)

    markdown, removed_count = _validate_markdown_links(
        markdown,
        trusted_urls,
    )

    if removed_count:
        logger.warning(
            "Output guardrails removed %d unsupported external link(s).",
            removed_count,
        )

    # Report structure issues. Keep the output available for review.
    issues = _check_expected_structure(markdown, state)

    for issue in issues:
        logger.warning("Output structure validation: %s", issue)

    logger.info(
        "Output guardrails completed: characters=%d, "
        "trusted_evidence_urls=%d, structure_issues=%d",
        len(markdown),
        len(trusted_urls),
        len(issues),
    )

    return markdown.strip() + "\n"


def merge_content(state: State) -> dict:
    """Merge worker sections in task order and validate the result."""
    plan = state.get("plan")

    if plan is None:
        raise ValueError("merge_content called without plan.")

    sections = state.get("sections", [])

    ordered_sections = [
        markdown
        for _, markdown in sorted(
            sections,
            key=lambda section: section[0],
        )
        if isinstance(markdown, str) and markdown.strip()
    ]

    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    validated_markdown = _apply_output_guardrails(
        merged_md,
        state,
    )

    return {"merged_md": validated_markdown}


__all__ = [
    "merge_content",
    "decide_images",
    "generate_and_place_images",
]
