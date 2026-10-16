
"""Reliable asynchronous Tavily research agent for BlogForge.

External search results are treated as untrusted data. Suspicious
instruction patterns are logged, and a trust-boundary formatter is
provided for downstream LLM prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Optional

from pydantic import ValidationError

from guardrails.prompt_injection import (
    scan_untrusted_content,
    wrap_untrusted_content,
)
from workflow.state import EvidenceItem, State

logger = logging.getLogger("blogforge")

MAX_QUERIES = 3
RESULTS_PER_QUERY = 2

MAX_TITLE_CHARS = 160
MAX_SNIPPET_CHARS = 350
MAX_URL_CHARS = 500


# ============================================================
# Text normalization
# ============================================================

def _truncate(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None

    text = " ".join(str(value).split()).strip()
    if not text:
        return None

    if len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0] + "..."

    return text


# ============================================================
# Prompt-injection scanning
# ============================================================

def _scan_result_for_injection(item: dict[str, Any]) -> None:
    """Log suspicious patterns without discarding potentially useful evidence."""

    fields = {
        "title": item.get("title"),
        "snippet": item.get("snippet"),
    }

    for field_name, content in fields.items():
        if not isinstance(content, str) or not content:
            continue

        result = scan_untrusted_content(content)

        if result.suspicious:
            logger.warning(
                "Potential prompt injection in Tavily result: "
                "url=%r field=%s categories=%s",
                item.get("url"),
                field_name,
                ", ".join(result.matched_categories),
            )


def format_evidence_for_prompt(
    evidence: list[EvidenceItem],
) -> str:
    """Format evidence behind an explicit untrusted-data boundary.

    Call this when research evidence is inserted into an LLM prompt.
    Do not interpolate raw search snippets directly into instructions.
    """

    serialized_items: list[dict[str, Any]] = []

    for item in evidence:
        if hasattr(item, "model_dump"):
            data = item.model_dump(mode="json")
        elif isinstance(item, dict):
            data = item
        else:
            data = {"content": str(item)}

        serialized_items.append(data)

    evidence_json = json.dumps(
        serialized_items,
        ensure_ascii=False,
        default=str,
    )

    return wrap_untrusted_content(
        evidence_json,
        source="Tavily research results",
    )


# ============================================================
# Tavily search
# ============================================================

def _tavily_search(
    query: str,
    max_results: int = RESULTS_PER_QUERY,
) -> list[dict[str, Any]]:
    """Run a blocking Tavily search; called through asyncio.to_thread."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing from .env.")

    from langchain_tavily import TavilySearch

    tool = TavilySearch(
        max_results=max_results,
        tavily_api_key=api_key,
    )

    raw = tool.invoke({"query": query})

    if isinstance(raw, dict):
        raw = raw.get("results", [])

    results: list[dict[str, Any]] = []

    for item in raw or []:
        if not isinstance(item, dict):
            continue

        url = _truncate(item.get("url"), MAX_URL_CHARS)
        if not url or not url.startswith(("https://", "http://")):
            continue

        normalized_item = {
            "title": _truncate(
                item.get("title") or item.get("name"),
                MAX_TITLE_CHARS,
            ) or "Untitled source",
            "url": url,
            "snippet": _truncate(
                item.get("content")
                or item.get("snippet")
                or item.get("description"),
                MAX_SNIPPET_CHARS,
            ),
            "published_at": _truncate(
                item.get("published_date")
                or item.get("published_at")
                or item.get("date"),
                40,
            ),
            "source": _truncate(
                item.get("source") or url,
                MAX_TITLE_CHARS,
            ),
        }

        # Scan as soon as external content has been normalized.
        _scan_result_for_injection(normalized_item)

        results.append(normalized_item)

    return results


async def _tavily_search_async(
    query: str,
    max_results: int = RESULTS_PER_QUERY,
) -> list[dict[str, Any]]:
    """Run Tavily without blocking the asyncio event loop."""

    return await asyncio.to_thread(
        _tavily_search,
        query,
        max_results,
    )


# ============================================================
# Date handling and deduplication
# ============================================================

def _iso_to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _deduplicate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []

    for item in results:
        url = item.get("url")
        if not url or url in seen:
            continue

        seen.add(url)
        unique.append(item)

    return unique


# ============================================================
# LangGraph research node
# ============================================================

async def research_node(state: State) -> dict:
    """Search Tavily concurrently and validate evidence."""

    raw_queries = state.get("queries", [])
    if not isinstance(raw_queries, list):
        raw_queries = []

    queries = [
        str(query).strip()
        for query in raw_queries
        if isinstance(query, str) and query.strip()
    ][:MAX_QUERIES]

    if not queries:
        logger.info("Research skipped: no search queries supplied.")
        return {"evidence": []}

    logger.info(
        "Starting %d Tavily searches, up to %d results per query.",
        len(queries),
        RESULTS_PER_QUERY,
    )

    search_results = await asyncio.gather(
        *(
            _tavily_search_async(query, RESULTS_PER_QUERY)
            for query in queries
        ),
        return_exceptions=True,
    )

    raw_results: list[dict[str, Any]] = []

    for query, result in zip(queries, search_results):
        if isinstance(result, BaseException):
            logger.warning(
                "Search failed for %r: %s",
                query,
                result,
            )
            continue

        raw_results.extend(result)

    unique_results = _deduplicate_results(raw_results)

    # Apply the existing freshness filter.
    mode = state.get("mode", "closed_book")

    if mode == "open_book":
        try:
            reference_date = date.fromisoformat(state["as_of"])
        except (KeyError, TypeError, ValueError):
            reference_date = date.today()

        try:
            recency_days = max(
                1,
                int(state.get("recency_days", 7)),
            )
        except (TypeError, ValueError):
            recency_days = 7

        cutoff = reference_date - timedelta(days=recency_days)

        filtered_results = []

        for item in unique_results:
            published = _iso_to_date(item.get("published_at"))

            # Preserve unknown publication dates, but log uncertainty.
            if published is None:
                logger.debug(
                    "Publication date unknown: %s",
                    item["url"],
                )
                filtered_results.append(item)
            elif published >= cutoff:
                filtered_results.append(item)

        unique_results = filtered_results

    # Validate the original evidence without embedding wrapper text
    # inside the EvidenceItem fields.
    evidence: list[EvidenceItem] = []

    for item in unique_results:
        try:
            evidence.append(EvidenceItem.model_validate(item))
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning(
                "Skipping invalid evidence item %r: %s",
                item.get("url"),
                exc,
            )

    logger.info(
        "Research completed with %d evidence items.",
        len(evidence),
    )

    return {"evidence": evidence}
