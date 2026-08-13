"""
utils/logging_config.py — Module 5.10 / 7.2 reliability guardrail.

Central logger so every LLM call and tool call in the pipeline logs
successes/failures to both console and a rotating file under outputs/logs/.
Import `get_logger(__name__)` from any script instead of using bare print()
for error paths.
"""

import logging
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "outputs", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "pipeline.log")

_configured = False


def _configure_root():
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
