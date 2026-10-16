
"""Shared asynchronous Groq LLM client for BlogForge."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def create_llm() -> ChatGroq:
    """Create the shared Groq chat model."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local .env file."
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )

    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_retries=2,
    )


# Shared model instance used by agents and workflow nodes.
llm = create_llm()
