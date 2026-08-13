"""
01_ingest_chunk.py — Module 4.2 / 4.3: Ingestion + Chunking

GOAL: Load all .txt files from your chosen scenario's data folder, split each into
overlapping chunks, and save the chunks (with metadata) to a JSON file for the next
stage to consume.

Expected output: outputs/chunks.json
    A list of {"id": ..., "source": ..., "text": ...} objects.

Time budget: ~15 minutes
"""

import os
import re
import json
import glob
from config import DATA_DIR, SCENARIO
from utils.logging_config import get_logger
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logger = get_logger(__name__)

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "chunks.json")

# --- Tunable chunking parameters (Module 4.3) ---
CHUNK_SIZE = 400        # characters per chunk (target size for both strategies)
CHUNK_OVERLAP = 80      # characters of overlap between consecutive chunks (fixed-size only)

# Which strategy to use for the "official" pipeline output (chunks.json).
# "sentence" is the stronger default — it never cuts a sentence in half.
# "fixed" is kept for the before/after comparison documented in RESULTS.md.
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "sentence").lower()


def load_documents(data_dir: str) -> list[dict]:
    """Load every .txt file in data_dir. Returns [{"source": filename, "text": content}]."""
    docs = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(filepath), "text": text})
    return docs


def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple fixed-size chunking with overlap (Module 4.3) — the beginner-tier
    strategy. Fast and simple, but can cut sentences (even words) in half at
    chunk boundaries. Kept here for the before/after comparison in RESULTS.md.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def sentence_aware_chunk(text: str, chunk_size: int) -> list[str]:
    """
    Paragraph- and sentence-aware chunking (Module 4.3 stronger strategy).

    Splits on blank lines first (paragraphs / numbered sections in these policy
    docs are separated by blank lines), then greedily packs whole sentences into
    chunks up to ~chunk_size characters. Never cuts a sentence or a word in half.
    A one-sentence "tail" overlap is carried into the next chunk so retrieval
    doesn't lose context right at a boundary.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para.replace("\n", " ").strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= chunk_size or not current:
                current = candidate
            else:
                chunks.append(current)
                # carry the last sentence forward as a light overlap for context continuity
                current = sentence
    if current:
        chunks.append(current)
    return chunks


def build_chunk_records(docs: list[dict], strategy: str = CHUNKING_STRATEGY) -> list[dict]:
    """Turn documents into a flat list of chunk records with stable IDs and metadata."""
    records = []
    chunk_id = 0
    for doc in docs:
        if strategy == "fixed":
            chunks = fixed_size_chunk(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            chunks = sentence_aware_chunk(doc["text"], CHUNK_SIZE)
        for chunk_text in chunks:
            records.append({
                "id": f"chunk_{chunk_id:04d}",
                "source": doc["source"],
                "text": chunk_text,
                "chunking_strategy": strategy,
            })
            chunk_id += 1
    return records


def main():
    print(f"Scenario: {SCENARIO}")
    print(f"Loading documents from: {DATA_DIR}")
    logger.info("Ingest starting for scenario=%s data_dir=%s", SCENARIO, DATA_DIR)

    try:
        docs = load_documents(DATA_DIR)
        if not docs:
            raise RuntimeError(
                f"No .txt files found in {DATA_DIR}. Check SCENARIO in your .env "
                "(should be 'banking' or 'healthcare')."
            )
        print(f"Loaded {len(docs)} documents.")

        records = build_chunk_records(docs, strategy=CHUNKING_STRATEGY)
        print(f"Produced {len(records)} chunks using '{CHUNKING_STRATEGY}' strategy "
              f"(target chunk_size={CHUNK_SIZE}).")

        # Quick before/after comparison against the naive fixed-size strategy,
        # printed to console so it's easy to screenshot / paste into RESULTS.md.
        fixed_records = build_chunk_records(docs, strategy="fixed")
        print(f"[Comparison] fixed-size strategy would have produced "
              f"{len(fixed_records)} chunks (sentence-aware: {len(records)}).")

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved chunks to: {OUTPUT_PATH}")
        logger.info("Ingest complete: %d chunks written to %s", len(records), OUTPUT_PATH)

        # Quick sanity print
        print("\n--- Sample chunk ---")
        print(json.dumps(records[0], indent=2)[:500])
    except Exception:
        logger.exception("Ingest & chunk stage failed.")
        raise


if __name__ == "__main__":
    main()
