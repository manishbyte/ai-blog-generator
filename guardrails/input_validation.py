"""Input validation guardrails for BlogForge."""

from __future__ import annotations

import re
from dataclasses import dataclass

from guardrails.prompt_injection import scan_untrusted_content


# ============================================================
# Configuration
# ============================================================

MAX_TOPIC_LENGTH = 500
MIN_TOPIC_LENGTH = 3

# Permit ordinary whitespace, but reject unexpected control characters.
_ALLOWED_WHITESPACE = {"\t", "\n", "\r"}


# ============================================================
# Result and exception types
# ============================================================

@dataclass(frozen=True)
class InputValidationResult:
    """Outcome of validating a blog topic."""

    valid: bool
    topic: str
    reason: str | None = None


class InputValidationError(ValueError):
    """Raised when a user-provided blog topic is invalid."""


# ============================================================
# Input validation
# ============================================================

def validate_topic(topic: str) -> InputValidationResult:
    """Validate and normalize a blog topic before workflow execution.

    Checks input type, length, control characters, and known
    prompt-injection patterns. Pattern detection is heuristic and
    cannot guarantee detection of every malicious instruction.
    """

    # 1. Validate input type.
    if not isinstance(topic, str):
        return InputValidationResult(
            valid=False,
            topic="",
            reason="The blog topic must be a string.",
        )

    # 2. Normalize whitespace.
    normalized = re.sub(r"\s+", " ", topic).strip()

    # 3. Reject empty topics.
    if not normalized:
        return InputValidationResult(
            valid=False,
            topic="",
            reason="The blog topic cannot be empty.",
        )

    # 4. Enforce minimum length.
    if len(normalized) < MIN_TOPIC_LENGTH:
        return InputValidationResult(
            valid=False,
            topic=normalized,
            reason=(
                f"The topic must contain at least "
                f"{MIN_TOPIC_LENGTH} characters."
            ),
        )

    # 5. Enforce maximum length.
    if len(normalized) > MAX_TOPIC_LENGTH:
        return InputValidationResult(
            valid=False,
            topic="",
            reason=(
                f"The topic exceeds the maximum length of "
                f"{MAX_TOPIC_LENGTH} characters."
            ),
        )

    # 6. Reject null bytes and unsupported control characters.
    for character in normalized:
        if character == "\x00":
            return InputValidationResult(
                valid=False,
                topic="",
                reason="The topic contains an invalid null character.",
            )

        if ord(character) < 32 and character not in _ALLOWED_WHITESPACE:
            return InputValidationResult(
                valid=False,
                topic="",
                reason="The topic contains unsupported control characters.",
            )

        if ord(character) == 127:
            return InputValidationResult(
                valid=False,
                topic="",
                reason="The topic contains an unsupported control character.",
            )

    # 7. Detect known prompt-injection patterns.
    scan = scan_untrusted_content(normalized)

    if scan.suspicious:
        categories = ", ".join(scan.matched_categories)

        return InputValidationResult(
            valid=False,
            topic="",
            reason=(
                "The topic was rejected because it contains a suspected "
                "prompt-injection attempt. Please enter a blog topic that "
                "describes the subject you want to write about. "
                f"Detected categories: {categories}."
            ),
        )

    # 8. All checks passed.
    return InputValidationResult(
        valid=True,
        topic=normalized,
    )


# ============================================================
# Strict validation
# ============================================================

def require_valid_topic(topic: str) -> str:
    """Return the normalized topic or raise InputValidationError."""

    result = validate_topic(topic)

    if not result.valid:
        raise InputValidationError(
            result.reason or "Invalid blog topic."
        )

    return result.topic
