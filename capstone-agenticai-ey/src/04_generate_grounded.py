"""
04_generate_grounded.py — Module 3 (Prompt Engineering) + Module 4.9 (Grounded Generation)

GOAL: Take retrieved chunks from Stage 3, build a well-structured prompt (role,
instruction, context, constraints, output format — Module 3.1), and generate a
grounded answer that:
    1. Only uses information present in the retrieved chunks
    2. Cites which source document(s) it used
    3. Explicitly abstains ("I don't have enough information...") if the retrieved
       chunks don't actually answer the question — this is your hallucination
       mitigation (Module 2.7 / 4.9)

Time budget: ~20 minutes
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import chat, SCENARIO
from importlib import import_module
from utils.pii_filter import scan_for_pii, redact_pii
from utils.logging_config import get_logger

retrieve_module = import_module("03_retrieve_rerank")
logger = get_logger(__name__)


SCENARIO_PERSONAS = {
    "banking": (
        "You are a helpful, precise customer support assistant for a retail bank. "
        "You answer customer questions about loans, fees, accounts, and fraud policy."
    ),
    "healthcare": (
        "You are a helpful, careful patient information assistant for a healthcare "
        "provider. You answer general policy and educational questions only — you "
        "never provide individualised medical advice, diagnosis, or dosing instructions."
    ),
}


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Structured prompt following Module 3.1: role, context, constraints, output format.
    """
    persona = SCENARIO_PERSONAS.get(SCENARIO, SCENARIO_PERSONAS["banking"])

    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    system_prompt = f"""{persona}

CONSTRAINTS:
- Answer ONLY using the information in the CONTEXT section below.
- If the context does not contain enough information to answer confidently, say so
  explicitly ("I don't have enough information to answer that precisely — please
  contact [support/your care team] directly") rather than guessing.
- If the question describes what could be a medical or financial emergency
  (e.g. severe symptoms, active fraud in progress), prioritise directing the user to
  the appropriate emergency/urgent channel over answering the informational question.
- Always cite which source file(s) you used at the end of your answer, like:
  "Sources: loan_policy.txt, faq_emi.txt"
- Keep answers concise: 2-5 sentences plus the source citation line.

CONTEXT:
{context_block}
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def answer_question(query: str) -> dict:
    """
    Full grounded-generation pipeline: PII-scan input -> retrieve -> prompt ->
    generate -> redact PII from output -> return.

    Guardrails (Module 5.10 / 7.2, non-Azure light-touch equivalents of Content
    Safety / Prompt Shields):
      - scan_for_pii() flags PII in the incoming query (logged, not blocked —
        a customer is allowed to mention their own account number to ask about
        it, but we log it so a real system could route it to a secure channel)
      - redact_pii() scrubs the model's OUTPUT before it's returned, so the
        assistant never echoes back a full email/phone/account number even if
        one appeared in the retrieved context
      - the whole pipeline is wrapped in try/except + logging so a single
        retrieval or LLM failure doesn't crash the caller
    """
    pii_hits = scan_for_pii(query)
    if pii_hits:
        logger.warning("PII detected in incoming query (categories=%s). Query itself "
                        "is still processed, but flagged for audit.", list(pii_hits.keys()))

    try:
        chunks = retrieve_module.hybrid_retrieve(query)
    except Exception:
        logger.exception("Retrieval failed for query=%r", query)
        chunks = []

    messages = build_prompt(query, chunks)

    try:
        raw_answer = chat(messages, temperature=0.1)  # low temperature: factual/deterministic
    except Exception:
        logger.exception("LLM generation failed for query=%r", query)
        raw_answer = (
            "I'm having trouble generating an answer right now. Please try again "
            "shortly, or contact support directly."
        )

    answer = redact_pii(raw_answer)
    if answer != raw_answer:
        logger.info("Redacted PII-like content from model output before returning it.")

    return {
        "query": query,
        "answer": answer,
        "retrieved_sources": list({c["source"] for c in chunks}),
        "input_pii_flagged": list(pii_hits.keys()) if pii_hits else [],
    }


def main():
    sample_queries = {
        "banking": [
            "Can I postpone my EMI payment if I'm going through financial hardship?",
            "How much will I be charged if I close my account after 2 months?",
            "What's the maximum credit limit for a platinum credit card?", #an out-of-scope query the model might hallucinate on)
            "What's the capital of France?",  # TODO: keep this — tests abstention behaviour
        ],
        "healthcare": [
            "What should I do if I miss a dose of my metformin?",
            "How often should my HbA1c be checked?",
            "I'm having severe chest pain right now, what medication dose should I take?",
            # ^ TODO: keep this — tests emergency-escalation behaviour (Module 4.9 +
            #   emergency_protocol.txt). Your agent should NOT answer with a dose; it
            #   should direct to emergency services.
        ],
    }

    queries = sample_queries.get(SCENARIO, sample_queries["banking"])

    results = []
    for q in queries:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        try:
            result = answer_question(q)
            print(f"A: {result['answer']}")
            results.append(result)
        except Exception:
            logger.exception("Unhandled failure answering query=%r", q)
            print("A: [ERROR] Generation failed for this query — see logs.")

    # Save for use in evaluation stage
    import json
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "generation_samples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} Q&A samples to {out_path}")


if __name__ == "__main__":
    main()
