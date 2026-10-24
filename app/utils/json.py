"""Robust JSON extraction for LLM output that may be wrapped in prose or code fences."""

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.+?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2026": "...",
}


class BadJSONError(ValueError):
    """Raised when no JSON object can be extracted from text."""


def _normalise(text: str) -> str:
    """Normalise typographic quotes that models emit inside JSON strings."""
    for bad, good in _SMART_QUOTES.items():
        text = text.replace(bad, good)
    return text


def _balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, respecting string literals.

    Args:
        text: Candidate text.

    Returns:
        str | None: The object substring, or None when braces are unbalanced.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    # Truncated output: close open braces so partial objects can still be salvaged
    tail = text[start:]
    closing = "}" * max(0, depth)
    if in_string:
        tail += '"'
    return tail + closing if closing else None


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from model output.

    Args:
        text: Raw model output (may contain prose, fences or trailing commentary).

    Returns:
        dict[str, Any]: Parsed object.

    Raises:
        BadJSONError: Nothing parseable was found.
    """
    if not text or not text.strip():
        raise BadJSONError("Empty response: expected a JSON object")

    candidates: list[str] = []
    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(text)

    for raw in candidates:
        candidate = _normalise(raw.strip())
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        block = _balanced_object(candidate)
        if not block:
            continue
        repaired = _TRAILING_COMMA.sub(r"\1", block)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise BadJSONError(f"No JSON object found in response: {text[:200].strip()}")


def extract_json_array(text: str) -> list[Any]:
    """Parse the first JSON array from model output.

    Args:
        text: Raw model output.

    Returns:
        list[Any]: Parsed array.

    Raises:
        BadJSONError: Nothing parseable was found.
    """
    if not text:
        raise BadJSONError("Empty response: expected a JSON array")
    fence = _FENCE.search(text)
    bodies = [fence.group(1)] if fence else []
    bodies.append(text)
    for body in bodies:
        candidate = _normalise(body.strip())
        start, end = candidate.find("["), candidate.rfind("]")
        if start == -1 or end <= start:
            continue
        chunk = candidate[start : end + 1]
        for attempt in (chunk, _TRAILING_COMMA.sub(r"\1", chunk)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue
    raise BadJSONError(f"No JSON array found in response: {text[:200].strip()}")
