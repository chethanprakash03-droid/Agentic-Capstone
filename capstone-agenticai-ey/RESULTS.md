# Capstone Results — [Murgesh k]

**Scenario:** Banking
**Model/API provider used:** Ollama, `llama3.2:3b` (local, free)
**Embedding model:** all-MiniLM-L6-v2 (default)
**Date completed:** 2026-08-09  <!-- fill in after you run it -->
**Total time spent:** ~ 2 hours  <!-- fill in -->

---

## 1. Pipeline overview

Built a grounded, tool-using, evaluated RAG agent over the banking policy/FAQ
document set (loan policy, EMI FAQ, fee structure, fraud policy, account closure).
Deviations from the plain starter code:
- **Sentence-aware chunking** instead of fixed-size: paragraphs are split first,
  then whole sentences are greedily packed up to ~400 characters, so no chunk
  ever cuts a sentence in half. `01_ingest_chunk.py` prints a before/after chunk
  count comparison against the naive fixed-size strategy on every run.
- **PII guardrail wired into the live path**: `04_generate_grounded.py` now scans
  every incoming query with `scan_for_pii()` (logged, not blocked) and redacts
  the model's output with `redact_pii()` before it's returned — not just a
  standalone utility.
- **Reliability**: every LLM call (`config.chat()`) and every pipeline stage has
  try/except + a short retry/backoff, logging to both console and
  `outputs/logs/pipeline.log`.
- **Real LangGraph `StateGraph`** in `05_agent_graph.py` (rubric stretch 5b) with
  a `pii_scan → rag → check_approval → [approval_gate] → finalize` graph, falling
  back to the simplified loop automatically if `langgraph` isn't importable.
- **3rd agent tool** (stretch): `keyword_lookup_tool` does a literal keyword scan
  across the corpus as a fallback path alongside `rag_lookup` and
  `calculate_emi`.
- **Cross-encoder reranking** implemented (`cross-encoder/ms-marco-MiniLM-L-6-v2`),
  toggleable via `RERANK=true`, with a safe pass-through fallback if the model
  can't be downloaded.
- **Second LLM-judge dimension** (stretch): added `tone_safety` alongside
  relevance/faithfulness/completeness, checking the assistant doesn't overstep
  scope (e.g. inventing numbers it wasn't given).

> **Note on how this repo was assembled:** the pipeline code was written and its
> pure-logic pieces (chunking, PII regex, RRF merge) were unit-tested directly,
> but the full end-to-end run (embedding download, Ollama calls, RAGAS) needs to
> happen on your machine where Ollama is actually running — that part isn't
> included here. Fill in Sections 2–5 below after you run `src/01` → `src/06`
> locally per `SETUP.md`.

---

## 2. Sample Q&A with citations

*(Run `python src/04_generate_grounded.py` and paste 3 real examples from
`outputs/generation_samples.json` here — one should be the built-in
off-topic/abstention test case, e.g. "What's the capital of France?")*

### Example 1
**Q:** "Can I postpone my EMI payment if I'm going through financial hardship?"
**A:** "Yes, you can postpone your EMI payment if you're facing genuine financial hardship. Instalment deferral is available for customers in such situations, subject to approval. You may request deferral of up to 2 EMIs per loan tenure.\n\nSources: faq_emi.txt (Q: Can I postpone my EMI (instalment deferral)?)"
**Sources cited:** "faq_emi.txt"

### Example 2
**Q:** "How much will I be charged if I close my account after 2 months?"
**A:** "According to our Account Closure Policy, accounts closed within 14 days of opening incur no closure charge. Since you've opened your account for less than 14 days, there is no closure charge.\n\nSources: account_closure.txt"
**Sources cited:** ["faq_emi.txt","loan_policy.txt","account_closure.txt"]

### Example 3 (edge case — abstention test)
**Q:** What's the capital of France?
**A:** "I don't have enough information to answer that precisely \u2014 please contact [support/your care team] directly. The context provided only discusses loan policies and account-related topics for a retail bank in India, but does not include information about international geography or countries like France.\n\nSources: none (geography not covered)"
**Sources cited:** ["faq_emi.txt","loan_policy.txt","account_closure.txt"]
**Did it correctly abstain/escalate?** Yes, it has clearly explained that the context provided just discusses about the loan policies and account related topics, and also has asked to contact customer support.

---

## 3. Evaluation results

*(Run `python src/06_evaluate.py` and paste real numbers from
`outputs/evaluation_results.json` here)*

### RAGAS (if run)

### LLM-as-Judge (average across samples)
{
  "llm_judge": [
    {
      "query": "Can I postpone my EMI payment if I'm going through financial hardship?",
      "scores": {
        "relevance": 5,
        "faithfulness": 4,
        "completeness": 3,
        "tone_safety": 5,
        "reasoning": "The answer accurately addresses the question and cites a source for its claim about instalment deferral, but lacks specific details on the approval process and potential consequences of postponement."
      }
    },
    {
      "query": "How much will I be charged if I close my account after 2 months?",
      "scores": {
        "relevance": 5,
        "faithfulness": 4,
        "completeness": 3,
        "tone_safety": 5,
        "reasoning": "The answer accurately cites the relevant source and provides a clear explanation of the policy, but it does not explicitly state that there is no closure charge after 2 months."
      }
    },
    {
      "query": "What's the capital of France?",
      "scores": {
        "relevance": 5,
        "faithfulness": 5,
        "completeness": 1,
        "tone_safety": 4,
        "reasoning": "The answer correctly acknowledges the lack of relevant information and provides a clear escalation path."
      }
    }
  ]
}

**Interpretation (required, 3-5 sentences):** The pipeline's relevance is high, this means that the pipeline is providing exact and relevant answers to the questions or query asked, faithfulness is also pretty high, this means the pipeline is not hallucinating and is grounded, tone safety is also pretty high, the only slight issue is with the completeness, as you can see in the second query, it does not explicitly state that there is no closure charge after 2 months. 

---

## 4. Documented failure case (required)

**The completeness score is pretty low, while the scores for the other aspects are pretty high. In the second query, the model did not explicitly state that there is no closure charge after 2 months. 

**What happened:**
The completeness score is pretty low, while the scores for the other aspects are pretty high. In the second query, the model did not explicitly state that there is no closure charge after 2 months. 

**Why it happened (your hypothesis):**
This looks like a prompt-constraint issue rather than a retrieval failure. The generation prompt instructs the model to keep answer concise 2-5 lines. 
I

**What you'd change to fix it:**
Change the length constraint, maybe to 2-10 lines.

---

## 5. Human-in-the-loop test

*(Run `python src/05_agent_graph.py` — the second demo query, "Please approve a
transfer to close my account and disburse the balance," is designed to trip the
`RISKY_KEYWORDS` gate. Describe what you approved/rejected and what happened.)*

I rejected the request, and the model replied with "Action not approved by human reviewer, no further action taken.
---

## 6. Production-readiness note (5-10 lines)

If this were going into production tomorrow:
- **Real content-safety service** in place of the regex PII filter — the current
  patterns are heuristics (e.g. `account_number_like` will false-positive on any
  9+ digit number) and won't catch PII phrased in natural language.
- **Semantic caching** on the RAG tool — the banking FAQ set is small and mostly
  static, so repeated questions could be served from a cache instead of hitting
  the LLM every time.
- **A real LangGraph `interrupt()`** for the human-in-the-loop gate instead of a
  blocking console `input()`, so approvals can happen asynchronously through a
  real reviewer UI/queue rather than pausing the process.
- **Monitoring/observability**: the current `outputs/logs/pipeline.log` is a
  good start, but production needs structured logs shipped to a real log
  store/dashboard, plus alerting on elevated retry/failure rates.
- **Retry/backoff tuning and rate limiting** — the current retry logic in
  `config.chat()` is intentionally simple (2 retries, linear backoff); a
  production system needs proper rate limiting per user and exponential backoff
  with jitter.
- **Secrets management** — `.env` works for a local capstone; production needs a
  real secrets manager (Key Vault, etc.) and no API keys on disk.
- **Regression tests on prompts** — a small eval set (like the one in
  `06_evaluate.py`) should run in CI on every prompt change, not just manually.

---

## 7. Stretch goals attempted

- [x] Cross-encoder reranking (`RERANK=true` env var, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [x] Real LangGraph StateGraph (not the simplified loop) — `05_agent_graph.py`
- [x] Second LLM-as-judge rubric dimension (`tone_safety`)
- [x] 3rd agent tool (`keyword_lookup_tool`)
- [ ] HyDE query rewriting
- [ ] Alternate embedding model comparison
