"""
utils/pii_filter.py — Module 5.10 / 7.2 lightweight guardrail

This is a stand-in for Azure Content Safety / Prompt Shields, which you don't have
quota for. It's intentionally simple — a regex-based check you can call on both user
input and model output.

Usage:
    from utils.pii_filter import scan_for_pii, redact_pii

    hits = scan_for_pii(user_input)
    if hits:
        print(f"Warning: possible PII detected: {hits}")

    safe_text = redact_pii(model_output)

TODO (stretch): extend PATTERNS with more categories relevant to your scenario
(e.g. account numbers, patient IDs) and log every redaction event using the
`logging` module per Module 5.10's reliability guidance.
"""

import re

from utils.logging_config import get_logger

logger = get_logger(__name__)

PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "account_number_like": re.compile(r"\b\d{9,18}\b"),  # loose heuristic — flags long digit runs
    # Extended per capstone scenario (banking): common loan/application/card
    # reference ID formats used in these fictional policy docs, e.g. "LN-2024-88213".
    "loan_or_application_id": re.compile(r"\b[A-Z]{2,4}-\d{4}-\d{3,8}\b"),
    "card_number_like": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def scan_for_pii(text: str) -> dict:
    """Returns {category: [matches]} for any PII patterns found in text."""
    hits = {}
    for category, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            hits[category] = matches
    if hits:
        logger.info("PII scan found categories: %s", list(hits.keys()))
    return hits


def redact_pii(text: str) -> str:
    """Replaces detected PII with a [REDACTED_<CATEGORY>] placeholder."""
    redacted = text
    for category, pattern in PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{category.upper()}]", redacted)
    return redacted


if __name__ == "__main__":
    test_text = "Contact me at jane.doe@example.com or 555-123-4567, account 1234567890123."
    print("Original:", test_text)
    print("PII found:", scan_for_pii(test_text))
    print("Redacted:", redact_pii(test_text))
