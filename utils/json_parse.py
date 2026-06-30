"""Robust JSON extractor used as a fallback when structured-output parsing fails."""

import json
import re
import logging

logger = logging.getLogger("control")


def parse_json(raw_text: str, expected_keys: list[str]) -> dict | None:
    """Extract the first valid JSON object from raw_text and validate expected_keys.

    Strips markdown fences and surrounding prose before parsing.
    Returns None (never raises) on any failure, and logs the raw output.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()

    # Find the first '{' and match braces
    start = text.find("{")
    if start == -1:
        logger.warning("parse_json: no '{' found. raw=%r", raw_text[:300])
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    logger.warning("parse_json: JSONDecodeError %s. raw=%r", exc, raw_text[:300])
                    return None
                missing = [k for k in expected_keys if k not in obj]
                if missing:
                    logger.warning(
                        "parse_json: missing keys %s. raw=%r", missing, raw_text[:300]
                    )
                    return None
                return obj

    logger.warning("parse_json: unmatched braces. raw=%r", raw_text[:300])
    return None
