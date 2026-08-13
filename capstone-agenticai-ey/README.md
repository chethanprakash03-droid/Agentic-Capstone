# Capstone Project — Agentic AI Training Program

**Duration:** 2 hours (self-paced, individual)
**Type:** Individual submission, hosted on your **personal** GitHub account
**Builds on:** All 20 days / 7 modules of the training program + your in-class lab work

---

## 1. Why this capstone

Every hands-on lab you did in class (Modules 1–6) was a mini version of one piece of a
real GenAI/Agentic system — preprocessing, embeddings, RAG, prompting, agents, evaluation.

This capstone asks you to **stitch those pieces together into one working pipeline**,
end-to-end, on a dataset you choose. Nothing here is new conceptually — it is
integration, not invention.

| Module | What you learned | Where it shows up in this capstone |
|---|---|---|
| 1 — NLP Fundamentals | tokenisation, TF-IDF, BM25 | Keyword half of hybrid retrieval |
| 2 — GenAI Fundamentals | decoding params, context windows, hallucination | Model config + grounding checks |
| 3 — Prompt Engineering | structured prompts, few-shot, CoT, injection defence | `04_generate_grounded.py` |
| 4 — RAG | chunking, embeddings, hybrid search, reranking, grounded generation | `01`–`04` scripts |
| 5 — Agentic AI | tools, LangGraph, memory, human-in-the-loop | `05_agent_graph.py` |
| 6 — Evaluation | RAGAS, LLM-as-judge, retrieval metrics | `06_evaluate.py` |
| 7 — Productionisation | logging, guardrails, config management (Azure-specific parts skipped) | `utils/pii_filter.py`, logging throughout |

Azure-specific services (AI Foundry, AI Search, Content Safety, App Insights) are
**intentionally excluded** — quota is limited. Open-source / any-vendor equivalents are
used instead, and everything is provider-agnostic (see Section 4).

---

## 2. Your goal

Build a **grounded, tool-using, evaluated RAG agent** over a small document set, in one
of two scenarios (your choice):

- **Scenario A — Banking Support Agent**: answers customer questions using loan/FAQ/policy
  documents, can calculate EMI, and pauses for human approval before "approving" any
  loan-related action.
- **Scenario B — Healthcare Info Agent**: answers patient questions using clinical
  guideline documents, can look up a simple date/scheduling utility, and pauses for
  human approval before finalising any medication-related recommendation.

Both scenarios ship with a ready-to-use sample dataset in `/data/`. You do not need to
source your own documents unless you want to.

---

## 3. Pick your scenario

Open `/data/banking/` or `/data/healthcare/` and skim the files. Pick **one**. Everything
downstream (chunking, retrieval, prompts, agent tools, eval questions) should use that
scenario consistently. Note your choice at the top of your `RESULTS.md`.

---

## 4. Pick your model/API — whatever you have access to

You do **not** need Azure OpenAI. Use whichever of these you already have, in order of
"least setup friction first":

| Option | Cost | Setup effort | Notes |
|---|---|---|---|
| **Azure OpenAI** | Free if you have leftover class quota | Low | Use if your quota still works |
| **OpenAI API** (`gpt-4o-mini` etc.) | ~$0.01–0.05 for this whole capstone | Low | Just need an API key |
| **Anthropic Claude API** | Similarly cheap | Low | Same pattern as OpenAI |
| **Any other hosted LLM API** (Gemini, Groq, Mistral, etc.) | Varies, usually cheap/free tier | Low | The code is provider-agnostic |
| **Ollama (local, free)** | Free | Higher — needs local install + a small model (e.g. `llama3.2:3b`, `phi3`) | Best if you have no API budget at all; slower, but zero cost |

All starter code reads the provider from `.env` via `config.py` — **you only change
config, never the pipeline logic**. See `SETUP.md` for exact `.env` examples for each
option.

> Embeddings do **not** require an API call — `sentence-transformers` runs locally and
> free regardless of which LLM you pick, so retrieval always works even if your LLM
> quota is tight.

---

## 5. Time budget (fit inside 2 hours)

This is a guide, not a hard rule — reallocate as needed, but don't let one stage eat the
whole session.

| Stage | Script | Time | Priority |
|---|---|---|---|
| 0. Setup & sanity check | `SETUP.md` | 10 min | Required |
| 1. Ingest + chunk | `01_ingest_chunk.py` | 15 min | Required |
| 2. Embed + index | `02_embed_index.py` | 15 min | Required |
| 3. Hybrid retrieval + rerank | `03_retrieve_rerank.py` | 20 min | Required (rerank = stretch if short on time) |
| 4. Grounded generation w/ citations | `04_generate_grounded.py` | 20 min | Required |
| 5. Agent + tools + human-in-loop | `05_agent_graph.py` | 25 min | Required (2nd tool = stretch if short) |
| 6. Evaluation (RAGAS + judge) | `06_evaluate.py` | 15 min | Required (do at least one metric family) |
| 7. Screenshots + RESULTS.md + push | — | 10 min | Required |

**If you're running out of time:** skip reranking (Stage 3) and skip the second agent
tool (Stage 5) first — these are marked optional in the rubric. Do not skip evaluation;
it's worth the most points relative to effort.

---

## 6. Architecture (what you're building)

```
                         ┌─────────────────────────┐
   data/*.txt   ───────▶ │ 01 Ingest & Chunk        │
                         └───────────┬─────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │ 02 Embed & Index          │  (sentence-transformers
                         │    (Chroma / FAISS)       │   + BM25 index)
                         └───────────┬─────────────┘
                                     ▼
              query ───▶ ┌─────────────────────────┐
                         │ 03 Hybrid Retrieval (RRF) │
                         │    + optional reranker    │
                         └───────────┬─────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │ 04 Grounded Generation     │  (structured prompt,
                         │    + citations             │   abstain-if-no-evidence)
                         └───────────┬─────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │ 05 LangGraph Agent         │  tools: [RAG tool,
                         │    + Human-in-the-loop     │   calculator/date tool]
                         │      approval gate         │  pauses on risky action
                         └───────────┬─────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │ 06 Evaluation               │  RAGAS: faithfulness,
                         │   (RAGAS + LLM-as-judge)   │  answer relevance, etc.
                         └─────────────────────────┘
```

---

## 7. Deliverables

By the end you should have, all committed to **your own personal GitHub repo**:

1. Working code for all 6 stages (`starter_code/` filled in — rename folder to `src/`
   once done, or keep as-is)
2. `RESULTS.md` (use `outputs/RESULTS_TEMPLATE.md`) with:
   - Scenario + model/API you chose
   - Sample Q&A pairs with the agent's answers and citations
   - Your RAGAS / judge evaluation scores + a 3–4 sentence interpretation
   - One documented failure case (e.g., a hallucination, a bad retrieval, an agent
     mis-step) and what you'd do to fix it — this is required and shows real learning
   - Screenshots (see `outputs/screenshots/README.md` for exact list)
3. A short **architecture note** (5–10 lines) on what you'd change for production
   (this connects back to Module 7 — you don't need Azure, just reason about it:
   caching, guardrails, monitoring, cost, etc.)
4. `requirements.txt` reflecting anything you added
5. A clean `.env.example` (never commit your real `.env` / API keys!)

Full required file list is in `SUBMISSION_CHECKLIST.md` — that's what gets evaluated.

---

## 8. Guardrails you must keep in (light-touch, non-Azure)

Since Azure Content Safety / Prompt Shields aren't available, implement the minimal
open equivalents already stubbed in `utils/pii_filter.py`:

- Basic PII regex filter on inputs/outputs (emails, phone numbers, account numbers)
- An "abstain if no relevant context retrieved" rule in the generation prompt
  (Module 4.9 grounding pattern)
- Try/except + logging around every LLM and tool call (Module 5.10 reliability pattern)

These are ~20 minutes of work total and are part of the required rubric — don't skip.

---

## 9. Stretch goals (optional, extra credit)

Pick any if you have time left:

- Add a 3rd tool to the agent (e.g., a document-lookup-by-keyword tool)
- Add cross-encoder reranking (Module 4.8)
- Add a second LLM-as-judge rubric dimension (e.g., tone/safety, not just faithfulness)
- Try HyDE query rewriting (Module 4.7) and compare retrieval before/after
- Swap in a different embedding model and compare retrieval quality

---

## 10. Submission

See `SUBMISSION_CHECKLIST.md` for the exact steps and file list. In short:

1. Fork/copy this structure into a **new public repo on your personal GitHub**
   (not your company org) named `agentic-ai-capstone-<yourname>`
2. Commit your working code, data, RESULTS.md, and screenshots
3. Make sure `git clone` + `pip install -r requirements.txt` + running the scripts
   **actually works** for someone else — this is graded
4. Share the repo link as instructed by the training team

Good luck — this is meant to be the fun part. You've already done the hard conceptual
work in class; this is just proving you can wire it together.
