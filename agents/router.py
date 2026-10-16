"""Reliable JSON-mode router for BlogForge."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from llm.client import llm
from workflow.state import RouterDecision, State

logger = logging.getLogger("blogforge")

ROUTER_SYSTEM = """
You route technical blog topics before planning.

Return exactly one valid JSON object:
{
  "needs_research": true,
  "mode": "hybrid",
  "reason": "Brief explanation",
  "queries": ["Search query 1", "Search query 2", "Search query 3"],
  "max_results_per_query": 2
}

Rules:
- closed_book: evergreen concepts; needs_research=false; queries=[].
- hybrid: current examples, tools, models, or evidence are useful.
- open_book: latest news, weekly roundups, current pricing, or changing policy.
- hybrid and open_book require needs_research=true.
- Return no more than 3 focused queries when research is required.
- Always include every JSON key.
- Do not return Markdown fences or explanatory text outside JSON.
"""


def _parse_json(content: Any) -> dict:
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )

    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Router returned no JSON object.")

    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Router JSON must be an object.")
    return result


def _fallback_decision(topic: str) -> RouterDecision:
    """Use research for topics that explicitly sound time-sensitive."""
    text = topic.lower()
    volatile = (
        "latest", "today", "this week", "weekly", "recent news",
        "current price", "current pricing", "breaking news",
    )

    if any(term in text for term in volatile):
        return RouterDecision(
            needs_research=True,
            mode="open_book",
            reason="Fallback: the topic appears time-sensitive.",
            queries=[
                f"{topic} latest developments",
                f"{topic} recent announcements",
                f"{topic} recent research and analysis",
            ],
            max_results_per_query=2,
        )

    return RouterDecision(
        needs_research=False,
        mode="closed_book",
        reason="Fallback: treating the topic as an evergreen explanation.",
        queries=[],
        max_results_per_query=2,
    )


async def router_node(state: State) -> dict:
    topic = state["topic"]

    try:
        json_llm = llm.bind(
            response_format={"type": "json_object"},
            max_tokens=1200,
        )

        response = await json_llm.ainvoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    f"As-of date: {state.get('as_of', 'unknown')}\n"
                    "Choose the research mode and return the JSON object."
                )
            ),
        ])

        raw = _parse_json(response.content)

        needs_research = raw.get("needs_research", False)
        if not isinstance(needs_research, bool):
            needs_research = str(needs_research).lower() == "true"

        mode = raw.get("mode", "closed_book")
        if mode not in {"closed_book", "hybrid", "open_book"}:
            mode = "hybrid" if needs_research else "closed_book"

        if mode == "closed_book":
            needs_research = False
            queries = []
        else:
            needs_research = True
            raw_queries = raw.get("queries", [])
            if not isinstance(raw_queries, list):
                raw_queries = []

            queries = [
                str(query).strip()
                for query in raw_queries
                if str(query).strip()
            ][:3]

            if not queries:
                queries = [f"{topic} reliable recent sources"]

        decision = RouterDecision.model_validate({
            "needs_research": needs_research,
            "mode": mode,
            "reason": str(
                raw.get("reason") or "Selected based on topic freshness."
            ),
            "queries": queries,
            "max_results_per_query": 2,
        })

    except Exception:
        logger.exception("Router failed; using fallback decision.")
        decision = _fallback_decision(topic)

    recency_days = {
        "closed_book": 3650,
        "hybrid": 45,
        "open_book": 7,
    }[decision.mode]

    logger.info(
        "Router mode=%s, research=%s, query_count=%d",
        decision.mode,
        decision.needs_research,
        len(decision.queries),
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state.get("needs_research", False) else "orchestrator"