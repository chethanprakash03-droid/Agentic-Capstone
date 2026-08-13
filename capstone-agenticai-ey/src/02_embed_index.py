"""
02_embed_index.py — Module 4.4 / 4.5: Embeddings + Vector Indexing

GOAL: Embed every chunk from outputs/chunks.json using a local sentence-transformers
model (free, no API needed) and store them in a persistent ChromaDB collection. Also
build a BM25 index over the same chunks for hybrid search in the next stage.

Expected output:
    - A persistent ChromaDB collection at outputs/chroma_db/
    - outputs/bm25_corpus.json (tokenised corpus for BM25, saved for reuse)

Time budget: ~15 minutes
"""

import os
import json
import chromadb
from rank_bm25 import BM25Okapi
from config import get_embedding_model, SCENARIO
from utils.logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "outputs", "chunks.json")
CHROMA_DIR = os.path.join(BASE_DIR, "outputs", "chroma_db")
BM25_CORPUS_PATH = os.path.join(BASE_DIR, "outputs", "bm25_corpus.json")
COLLECTION_NAME = f"capstone_{SCENARIO}"


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_index(chunks: list[dict]):
    """Embed chunks and store them in a persistent Chroma collection."""
    model = get_embedding_model()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Reset collection if it already exists (idempotent reruns)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Vector index built: {CHROMA_DIR} (collection='{COLLECTION_NAME}')")
    return collection


def build_bm25_index(chunks: list[dict]):
    """
    Save a tokenised corpus for BM25 (Module 1.2 / 4.6). We rebuild the BM25Okapi
    object fresh in stage 03 from this saved corpus rather than pickling it directly,
    which keeps things simple and version-safe.

    TODO (optional improvement): use a real tokenizer (e.g. nltk.word_tokenize) instead
    of naive .split() for better keyword matching on punctuation-heavy text.
    """
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    payload = {
        "ids": [c["id"] for c in chunks],
        "tokenized_corpus": tokenized_corpus,
    }
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"BM25 corpus saved: {BM25_CORPUS_PATH}")

    # quick smoke test
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 index built OK ({len(tokenized_corpus)} docs).")


def main():
    try:
        chunks = load_chunks()
        print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

        build_vector_index(chunks)
        build_bm25_index(chunks)

        print("\nStage 2 complete. Both vector and keyword indexes are ready for retrieval.")
        logger.info("Embed & index stage complete for %d chunks.", len(chunks))
    except Exception:
        logger.exception("Embed & index stage failed.")
        raise


if __name__ == "__main__":
    main()
