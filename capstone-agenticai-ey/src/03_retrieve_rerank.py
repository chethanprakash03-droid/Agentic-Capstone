"""
03_retrieve_rerank.py — Module 4.6 / 4.8: Hybrid Search + Reranking

GOAL: Given a query, retrieve top-k chunks using BOTH vector similarity and BM25
keyword search, merge results using Reciprocal Rank Fusion (RRF), and (optional
stretch) rerank the merged results with a cross-encoder.

This mirrors your Module 1 Option C lab (BM25 vs semantic) and Module 4 Option C lab
(hybrid search) — same idea, just without Azure AI Search.

Time budget: ~20 minutes (skip reranking first if short on time — it's marked optional)
"""

import os
import json
import chromadb
from rank_bm25 import BM25Okapi
from config import get_embedding_model, SCENARIO
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Stretch goal (Module 4.8): set to True to enable cross-encoder reranking by
# default in hybrid_retrieve(). Set via env var so it can be toggled without
# editing code: RERANK=true python src/03_retrieve_rerank.py
USE_RERANKER = os.getenv("RERANK", "false").lower() == "true"
_reranker_model = None

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "outputs", "chunks.json")
CHROMA_DIR = os.path.join(BASE_DIR, "outputs", "chroma_db")
BM25_CORPUS_PATH = os.path.join(BASE_DIR, "outputs", "bm25_corpus.json")
COLLECTION_NAME = f"capstone_{SCENARIO}"

TOP_K_VECTOR = 5
TOP_K_BM25 = 5
TOP_K_FINAL = 4
RRF_K = 60  # standard RRF smoothing constant


def load_chunks_lookup() -> dict:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["id"]: c for c in chunks}


def vector_search(query: str, top_k: int = TOP_K_VECTOR) -> list[str]:
    """Returns a ranked list of chunk IDs from vector similarity search."""
    model = get_embedding_model()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["ids"][0]  # ranked list of chunk ids


def bm25_search(query: str, top_k: int = TOP_K_BM25) -> list[str]:
    """Returns a ranked list of chunk IDs from BM25 keyword search."""
    with open(BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    ids = payload["ids"]
    tokenized_corpus = payload["tokenized_corpus"]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_k]]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """
    Merge multiple ranked lists of chunk IDs using RRF (Module 4.6).
    score(doc) = sum over lists of 1 / (k + rank_in_that_list)
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in fused]


def hybrid_retrieve(query: str, top_k: int = TOP_K_FINAL, use_reranker: bool = None) -> list[dict]:
    """Full hybrid retrieval pipeline: vector + BM25 -> RRF -> (optional rerank) -> top-k."""
    if use_reranker is None:
        use_reranker = USE_RERANKER

    try:
        vector_ids = vector_search(query)
    except Exception:
        logger.exception("Vector search failed for query=%r; falling back to BM25 only.", query)
        vector_ids = []

    try:
        bm25_ids = bm25_search(query)
    except Exception:
        logger.exception("BM25 search failed for query=%r; falling back to vector only.", query)
        bm25_ids = []

    # Pull extra candidates before the final cut if we're going to rerank, so the
    # reranker has something meaningful to re-order rather than just re-scoring
    # the same top_k it would've returned anyway.
    fuse_k = top_k * 3 if use_reranker else top_k
    fused_ids = reciprocal_rank_fusion([vector_ids, bm25_ids])[:fuse_k]

    lookup = load_chunks_lookup()
    candidates = [lookup[cid] for cid in fused_ids if cid in lookup]

    if use_reranker and candidates:
        return rerank_cross_encoder(query, candidates, top_k=top_k)
    return candidates[:top_k]


def _get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def rerank_cross_encoder(query: str, chunks: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    STRETCH (Module 4.8): cross-encoder reranking. Re-scores each (query, chunk)
    pair with a cross-encoder — much more accurate than the bi-encoder similarity
    used in vector_search, at the cost of being slower, so it only runs over the
    already-fused candidate set rather than the whole corpus.

    Falls back to a pass-through (no reranking) if the cross-encoder model can't
    be loaded (e.g. no internet access to download it) — the pipeline should
    never hard-fail just because the stretch goal is unavailable.
    """
    try:
        reranker = _get_reranker_model()
        pairs = [[query, c["text"]] for c in chunks]
        scores = reranker.predict(pairs)
        reranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        return reranked[:top_k]
    except Exception:
        logger.exception(
            "Cross-encoder reranking failed/unavailable; falling back to RRF order "
            "(pass-through). Note this in RESULTS.md if it happens for you."
        )
        return chunks[:top_k]


def main():
    # A few sample queries per scenario — edit to match your chosen scenario
    sample_queries = {
        "banking": [
            "Can I postpone my EMI payment?",
            "What happens if I don't report a fraudulent transaction quickly?",
        ],
        "healthcare": [
            "What should I do if I miss a dose of my medication?",
            "How often should I get my HbA1c checked?",
        ],
    }

    queries = sample_queries.get(SCENARIO, sample_queries["banking"])
    print(f"Reranking enabled: {USE_RERANKER} (toggle with RERANK=true env var)\n")

    for query in queries:
        print(f"\n{'='*70}\nQuery: {query}\n{'='*70}")
        results = hybrid_retrieve(query)
        for r in results:
            preview = r["text"][:150].replace("\n", " ")
            print(f"[{r['source']}] {preview}...")


if __name__ == "__main__":
    main()
