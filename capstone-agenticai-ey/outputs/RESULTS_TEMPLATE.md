# Capstone Results — [Your Name]

**Scenario:** Banking / Healthcare *(delete one)*
**Model/API provider used:** e.g. OpenAI gpt-4o-mini / Anthropic claude-haiku-4-5 / Ollama llama3.2:3b
**Embedding model:** all-MiniLM-L6-v2 (default) / *(note if changed)*
**Date completed:** YYYY-MM-DD
**Total time spent:** ~ X hours Y minutes

---

## 1. Pipeline overview

Brief note (2-3 sentences) on what you built and any deviations from the default
starter code (e.g. "used sentence-aware chunking instead of fixed-size", "swapped
embedding model to X").

---

## 2. Sample Q&A with citations

### Example 1
**Q:** ...
**A:** ...
**Sources cited:** ...

### Example 2
**Q:** ...
**A:** ...
**Sources cited:** ...

### Example 3 (edge case — abstention or emergency escalation test)
**Q:** ...
**A:** ...
**Sources cited:** ...
**Did it correctly abstain/escalate?** Yes/No — explain

---

## 3. Evaluation results

### RAGAS (if run)
| Metric | Score |
|---|---|
| Faithfulness | |
| Answer Relevancy | |

### LLM-as-Judge (average across samples)
| Dimension | Avg Score (1-5) |
|---|---|
| Relevance | |
| Faithfulness | |
| Completeness | |

**Interpretation (required, 3-5 sentences):** What do these numbers actually tell you
about your pipeline? Where is it strong, where is it weak, and why do you think that is
(retrieval issue vs prompt issue vs model issue)?

---

## 4. Documented failure case (required)

Describe one thing that didn't work as expected — a hallucination, a bad retrieval, an
agent tool misfire, a guardrail gap, anything genuine.

**What happened:**
...

**Why it happened (your hypothesis):**
...

**What you'd change to fix it:**
...

---

## 5. Human-in-the-loop test

Describe the query that triggered your approval gate, what you approved/rejected, and
what happened as a result.

---

## 6. Production-readiness note (5-10 lines)

If this were going into production tomorrow, what's missing? (e.g. real content-safety
API instead of the regex PII filter, semantic caching, monitoring dashboards, retry/
backoff on API failures, rate limiting, proper secrets management, CI/CD regression
tests on prompts...)

---

## 7. Stretch goals attempted

- [ ] Cross-encoder reranking
- [ ] Real LangGraph StateGraph (not the simplified loop)
- [ ] Second LLM-as-judge rubric dimension
- [ ] HyDE query rewriting
- [ ] Alternate embedding model comparison
- [ ] Other: ___
