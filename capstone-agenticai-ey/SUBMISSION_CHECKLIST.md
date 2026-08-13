# Submission Checklist

## 1. Repo setup

- [ ] Create a **new public repository on your personal GitHub account**
      (NOT your company/organisation account — this is your individual capstone)
- [ ] Name it: `agentic-ai-capstone-<yourfirstname>` (e.g. `agentic-ai-capstone-priya`)
- [ ] Copy this entire folder structure into that repo
- [ ] Do **not** commit your real `.env` file or any API keys — only `.env.example`
      (this repo's `.gitignore` already excludes `.env`, `venv/`, `outputs/chroma_db/`
      — double check before your first commit)

## 2. Required files in your final repo

```
agentic-ai-capstone-<yourname>/
├── README.md                          (can keep the original brief, or personalise it)
├── SETUP.md
├── requirements.txt
├── .env.example
├── .gitignore
├── RESULTS.md                          <-- YOU fill this in (from outputs/RESULTS_TEMPLATE.md)
├── data/
│   └── banking/  or  healthcare/       (whichever scenario you picked — you can delete the other)
├── src/                                 <-- your completed code (renamed from starter_code/)
│   ├── config.py
│   ├── 01_ingest_chunk.py
│   ├── 02_embed_index.py
│   ├── 03_retrieve_rerank.py
│   ├── 04_generate_grounded.py
│   ├── 05_agent_graph.py
│   ├── 06_evaluate.py
│   └── utils/
│       └── pii_filter.py
├── outputs/
│   ├── chunks.json
│   ├── generation_samples.json
│   ├── evaluation_results.json
│   └── screenshots/
│       ├── 01_ingest_output.png
│       ├── 02_index_output.png
│       ├── 03_retrieval_output.png
│       ├── 04_generation_output.png
│       ├── 05_agent_hitl_output.png
│       └── 06_evaluation_output.png
└── evaluation_rubric.md               (leave as-is, for reference)
```

> Note: `outputs/chroma_db/` (the vector database files) can be large/binary — it's
> fine to exclude it via `.gitignore` as long as `outputs/chunks.json` and the JSON
> outputs are present, since the grader will regenerate the index by running your
> scripts anyway.

## 3. Screenshots — exact list required

Take a screenshot of your **terminal output** for each stage (not the code, the
*running output*). Save into `outputs/screenshots/` using the exact filenames above.

- `01_ingest_output.png` — showing chunk count + sample chunk printed
- `02_index_output.png` — showing "Vector index built" + "BM25 index built OK"
- `03_retrieval_output.png` — showing at least one query with its retrieved chunks
- `04_generation_output.png` — showing at least one grounded answer with citation,
  AND the abstention/off-topic test case handled correctly
- `05_agent_hitl_output.png` — showing the human-in-the-loop approval prompt firing
  and being approved/rejected
- `06_evaluation_output.png` — showing your RAGAS and/or LLM-judge scores printed

## 4. RESULTS.md — required content

Use `outputs/RESULTS_TEMPLATE.md` as your starting point. It must include:

- [ ] Scenario chosen (banking or healthcare)
- [ ] Model/API provider used
- [ ] At least 3 sample Q&A pairs with citations
- [ ] Evaluation scores (RAGAS and/or LLM-judge) with a written interpretation
      (not just numbers — what do they tell you about your pipeline?)
- [ ] **One documented failure case** — something that didn't work perfectly
      (a hallucination, a bad retrieval, an agent mis-step, a guardrail that let
      something through) and what you'd change to fix it. This is required, not
      optional — it's how we know you actually tested critically rather than just
      running the happy path once.
- [ ] Short production-readiness note (5–10 lines): what would you add before this
      goes live (caching, monitoring, real content-safety service, etc.)?
- [ ] Which stretch goals (if any) you attempted

## 5. Reproducibility check — do this yourself before submitting

Simulate being the grader:

```bash
# In a fresh terminal / fresh clone of your own repo:
git clone <your-repo-url>
cd agentic-ai-capstone-<yourname>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your own key again to test
python src/config.py               # sanity check
python src/01_ingest_chunk.py
python src/02_embed_index.py
python src/03_retrieve_rerank.py
python src/04_generate_grounded.py
python src/05_agent_graph.py
python src/06_evaluate.py
```

If any step fails on a clean clone, fix it before submitting — this is explicitly part
of the grading criteria (Module 7 production-readiness thinking applies to your own
repo, not just the app).

## 6. Final step

Share your repo URL as instructed by the training team.
