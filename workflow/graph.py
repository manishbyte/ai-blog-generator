
"""Main LangGraph workflow for BlogForge."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from langgraph.graph import StateGraph, START, END

from agents.router import router_node, route_next
from agents.research import research_node
from agents.orchestrator import orchestrator_node
from agents.worker import run_workers

from workflow.state import State
from workflow.reducer import (
    merge_content,
    decide_images,
    generate_and_place_images,
)


# ============================================================
# Concurrent worker node
# ============================================================

async def workers_node(state: State) -> dict:
    """Run all planned section workers concurrently."""

    plan = state.get("plan")

    if plan is None:
        raise ValueError("workers_node called without a plan.")

    sections = await run_workers(
        plan=plan,
        topic=state["topic"],
        mode=state["mode"],
        as_of=state["as_of"],
        recency_days=state["recency_days"],
        evidence=state.get("evidence", []),
    )

    return {"sections": sections}


# ============================================================
# Reducer subgraph
# ============================================================

def build_reducer_graph():
    """Build the section-merging and optional image-generation graph."""

    reducer = StateGraph(State)

    reducer.add_node("merge_content", merge_content)
    reducer.add_node("decide_images", decide_images)
    reducer.add_node(
        "generate_and_place_images",
        generate_and_place_images,
    )

    reducer.add_edge(START, "merge_content")
    reducer.add_edge("merge_content", "decide_images")
    reducer.add_edge("decide_images", "generate_and_place_images")
    reducer.add_edge("generate_and_place_images", END)

    return reducer.compile()


reducer_graph = build_reducer_graph()


# ============================================================
# Main workflow
# ============================================================

def build_graph():
    """Build and compile the complete BlogForge workflow."""

    workflow = StateGraph(State)

    workflow.add_node("router", router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("workers", workers_node)
    workflow.add_node("reducer", reducer_graph)

    # Begin with the router.
    workflow.add_edge(START, "router")

    # Research only when the router determines it is necessary.
    workflow.add_conditional_edges(
        "router",
        route_next,
        {
            "research": "research",
            "orchestrator": "orchestrator",
        },
    )

    # Both paths proceed to planning.
    workflow.add_edge("research", "orchestrator")

    # Plan -> concurrent section workers -> reducer.
    workflow.add_edge("orchestrator", "workers")
    workflow.add_edge("workers", "reducer")
    workflow.add_edge("reducer", END)

    return workflow.compile()


graph = build_graph()


# ============================================================
# Public async entry point
# ============================================================

async def generate_blog(
    topic: str,
    *,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Generate a blog asynchronously and return the final state."""

    cleaned_topic = topic.strip()

    if not cleaned_topic:
        raise ValueError("Topic cannot be empty.")

    run_date = as_of or date.today().isoformat()

    initial_state: State = {
        "topic": cleaned_topic,
        "as_of": run_date,
        "mode": "closed_book",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "recency_days": 3650,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    result = await graph.ainvoke(initial_state)

    return result
