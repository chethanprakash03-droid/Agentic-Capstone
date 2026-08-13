"""
06_evaluate.py — Module 6: Evaluation (RAGAS + LLM-as-Judge)

GOAL: Evaluate the quality of your grounded generation pipeline using:
  1. RAGAS metrics: faithfulness, answer relevancy (Module 6.5)
  2. A simple custom LLM-as-judge rubric (Module 6.7): relevance, faithfulness,
     completeness on a 1-5 scale

This reads outputs/generation_samples.json (produced by Stage 4) — run that first.

Time budget: ~15 minutes. If RAGAS gives you dependency trouble, the LLM-as-judge
section alone is an acceptable minimum for this stage — note that in RESULTS.md.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(__file__))

from config import chat, SCENARIO
from importlib import import_module
from utils.logging_config import get_logger

retrieve_module = import_module("03_retrieve_rerank")
logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAMPLES_PATH = os.path.join(BASE_DIR, "outputs", "generation_samples.json")
EVAL_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "evaluation_results.json")


# ---------------------------------------------------------------------------
# Part A: RAGAS evaluation (Module 6.5)
# ---------------------------------------------------------------------------

def run_ragas_evaluation(samples: list[dict]) -> dict:
    """
    Runs RAGAS faithfulness + answer_relevancy over the saved Q&A samples.

    TODO: RAGAS needs `contexts` (list of retrieved chunk texts) per question, which
    we re-fetch here via hybrid_retrieve for consistency with what was actually used.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    except ImportError as e:
        print(f"[SKIP] RAGAS not available or failed to import: {e}")
        return {}

    rows = []
    for s in samples:
        chunks = retrieve_module.hybrid_retrieve(s["query"])
        rows.append({
            "question": s["query"],
            "answer": s["answer"],
            "contexts": [c["text"] for c in chunks],
        })

    dataset = Dataset.from_list(rows)

    try:
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        print("\n--- RAGAS Results ---")
        print(result)
        return dict(result)
    except Exception as e:
        print(f"[FAIL] RAGAS evaluation run failed: {e}")
        print("       This is a known RAGAS dependency quirk sometimes — it's OK to")
        print("       note this in RESULTS.md and rely on the LLM-as-judge scores below.")
        return {}


# ---------------------------------------------------------------------------
# Part B: LLM-as-judge (Module 6.7) — simple, self-contained, no extra deps
# ---------------------------------------------------------------------------

# Second rubric dimension added as a stretch goal (Section 9 bullet 3):
# "tone_safety" checks the answer doesn't overstep scope (e.g. giving a bank a
# specific medical dose, or a healthcare bot giving individualised financial
# advice) and stays appropriately cautious/professional in tone.
JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge for a customer-facing AI assistant.

Rate the ANSWER to the QUESTION on four dimensions, each on a 1-5 scale
(1=poor, 5=excellent):
- relevance: does the answer address what was actually asked?
- faithfulness: is the answer fully supported by the SOURCES used (no invented facts)?
- completeness: does the answer cover what a user would need to know?
- tone_safety: is the tone professional/appropriate, and does the answer correctly
  abstain or escalate instead of overstepping its scope (e.g. not inventing a medical
  dosage, not promising a financial outcome it can't guarantee)?

QUESTION: {question}
ANSWER: {answer}
SOURCES CITED: {sources}

Respond in this exact JSON format, nothing else:
{{"relevance": <int>, "faithfulness": <int>, "completeness": <int>, "tone_safety": <int>, "reasoning": "<one sentence>"}}
"""


def llm_judge(sample: dict) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=sample["query"],
        answer=sample["answer"],
        sources=", ".join(sample.get("retrieved_sources", [])),
    )
    try:
        response = chat([{"role": "user", "content": prompt}], temperature=0)
    except Exception:
        logger.exception("LLM-as-judge call failed for query=%r", sample["query"])
        return {"relevance": None, "faithfulness": None, "completeness": None,
                "tone_safety": None, "reasoning": "llm_call_failed"}

    try:
        # Strip markdown code fences if the model added them
        cleaned = response.strip().strip("`").replace("json", "", 1).strip()
        scores = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse judge response as JSON: %s", response)
        print(f"[WARN] Could not parse judge response as JSON: {response}")
        scores = {"relevance": None, "faithfulness": None, "completeness": None,
                   "tone_safety": None, "reasoning": "parse_failed"}
    return scores


def run_llm_judge_evaluation(samples: list[dict]) -> list[dict]:
    results = []
    for s in samples:
        scores = llm_judge(s)
        results.append({"query": s["query"], "scores": scores})
        print(f"\nQ: {s['query']}\nScores: {scores}")
    return results


# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(SAMPLES_PATH):
        raise RuntimeError(
            f"{SAMPLES_PATH} not found. Run 04_generate_grounded.py first to "
            "produce generation samples."
        )

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} generation samples for evaluation.\n")

    try:
        print("=" * 70)
        print("PART A: RAGAS EVALUATION")
        print("=" * 70)
        ragas_results = run_ragas_evaluation(samples)

        print("\n" + "=" * 70)
        print("PART B: LLM-AS-JUDGE EVALUATION")
        print("=" * 70)
        judge_results = run_llm_judge_evaluation(samples)

        output = {"ragas": ragas_results, "llm_judge": judge_results}
        with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved evaluation results to {EVAL_OUTPUT_PATH}")
        print("\n>>> Copy your key scores + a short interpretation into RESULTS.md now. <<<")
        logger.info("Evaluation stage complete. Results at %s", EVAL_OUTPUT_PATH)
    except Exception:
        logger.exception("Evaluation stage failed.")
        raise


if __name__ == "__main__":
    main()
