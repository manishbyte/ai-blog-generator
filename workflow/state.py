
"""Shared schemas and LangGraph state for BlogForge.

Schemas are preserved from the original blog writer implementation.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ============================================================
# 1. Planning schemas
# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should do/understand.",
    )
    
    bullets: List[str] = Field(..., min_length=3, max_length=6)

    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


# ============================================================
# 2. Research schemas
# ============================================================

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None



class RouterDecision(BaseModel):
    needs_research: bool = Field(
        description="Whether external web research is needed."
    )
    mode: Literal["closed_book", "hybrid", "open_book"] = Field(
        description=(
            "closed_book for evergreen topics; hybrid for topics needing "
            "some current evidence; open_book for volatile news or latest updates."
        )
    )
    reason: str = Field(
        default="Research routing decision based on topic freshness and evidence needs.",
        description="Briefly explain why this research mode was selected."
    )
    queries: List[str] = Field(
        default_factory=list,
        description="Up to 3 focused web search queries when research is needed."
    )
    max_results_per_query: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum search results requested per query."
    )



class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ============================================================
# 3. Image planning schemas
# ============================================================

class ImageSpec(BaseModel):
    placeholder: str = Field(
        ...,
        description="e.g. [[IMAGE_1]]",
    )
    filename: str = Field(
        ...,
        description="Save under images/, e.g. qkv_flow.png",
    )
    alt: str
    caption: str
    prompt: str = Field(
        ...,
        description="Prompt to send to the image model.",
    )
    size: Literal[
        "1024x1024",
        "1024x1536",
        "1536x1024",
    ] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


# ============================================================
# 4. Shared LangGraph state
# ============================================================

class State(TypedDict, total=False):
    topic: str

    # Routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # Recency
    as_of: str
    recency_days: int

    # Workers: preserve the original append reducer
    sections: Annotated[List[tuple[int, str]], operator.add]

    # Reducer / image flow
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    # Final output
    final: str
