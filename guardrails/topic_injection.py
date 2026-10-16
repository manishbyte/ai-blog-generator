
"""High-confidence prompt-injection detection for BlogForge topics."""

import re


INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(
        r"\b(ignore|disregard|override|forget)\b"
        r".{0,100}\b(previous|prior|above|all)\b"
        r".{0,60}\b(instructions?|rules?|prompts?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "secret_disclosure": re.compile(
        r"\b(reveal|print|expose|show|dump|exfiltrate)\b"
        r".{0,100}\b(system prompt|hidden instructions?|"
        r"api keys?|environment variables?|database credentials?|"
        r"secrets?|private data)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "safety_bypass": re.compile(
        r"\b(disable|bypass|turn off|circumvent)\b"
        r".{0,80}\b(guardrails?|safety checks?|security rules?|"
        r"security controls?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "unauthorized_access": re.compile(
        r"\b(execute commands?|access private data|"
        r"steal credentials?|dump secrets?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
}


def detect_topic_injection(topic: str) -> str | None:
    """Return a detection category for suspicious topics, otherwise None."""
    if not isinstance(topic, str):
        return "invalid_input"

    matches = [
        category
        for category, pattern in INJECTION_PATTERNS.items()
        if pattern.search(topic)
    ]

    # Block direct override/disclosure attempts, or multiple attack signals.
    if (
        "instruction_override" in matches
        or "secret_disclosure" in matches
        or "safety_bypass" in matches
        or len(matches) >= 2
    ):
        return ", ".join(matches)

    return None
