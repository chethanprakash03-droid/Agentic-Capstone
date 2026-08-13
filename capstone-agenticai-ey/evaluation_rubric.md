# Evaluation Rubric — 100 points total

Scored per submission by the training team. Partial credit applies throughout —
a working "minimum viable" pipeline with honest documentation of what's missing scores
better than an incomplete pipeline with no explanation.

| # | Criterion | Module(s) | Points | What "full credit" looks like |
|---|---|---|---|---|
| 1 | **Ingestion & chunking** runs, produces sensible chunks | 4.2, 4.3 | 10 | `01_ingest_chunk.py` runs cleanly, `chunks.json` has reasonable chunk boundaries (not cutting words/sentences awkwardly at scale) |
| 2 | **Embedding & indexing** works, both vector + BM25 built | 4.4, 4.5, 1.2 | 10 | `02_embed_index.py` runs, Chroma collection + BM25 corpus both exist and are queryable |
| 3 | **Hybrid retrieval** returns relevant chunks for sample queries | 4.6, 1.1–1.2 | 15 | RRF merge implemented correctly; retrieved chunks are actually relevant to the 2+ sample queries tested |
| 3b | *(Stretch, +5)* Reranking implemented and compared | 4.8 | +5 | Cross-encoder reranking added, before/after comparison documented in RESULTS.md |
| 4 | **Grounded generation** with structured prompt, citations, and abstention | 3.1–3.3, 4.9, 2.7 | 20 | Prompt has clear role/context/constraints/format; answers cite sources; the "no relevant info" and "emergency/off-topic" test queries are handled correctly (abstains or escalates, doesn't hallucinate) |
| 5 | **Agent with tools + human-in-the-loop** | 5.1–5.4, 5.7 | 20 | At least 2 tools working (RAG tool + 1 utility tool); approval gate correctly triggers on the risky-keyword test case and blocks/allows based on response |
| 5b | *(Stretch, +5)* Real LangGraph StateGraph (not the simplified loop) | 5.6 | +5 | Skeleton in `05_agent_graph.py` implemented as an actual compiled `StateGraph` |
| 6 | **Evaluation** — RAGAS and/or LLM-as-judge run with interpreted results | 6.1, 6.3, 6.5, 6.7 | 15 | At least one evaluation method runs successfully with real scores (not placeholders) AND RESULTS.md includes a genuine interpretation, not just raw numbers |
| 7 | **Guardrails / reliability basics** | 5.10, 7.2 (non-Azure) | 5 | PII filter integrated somewhere in the flow (input or output); try/except + logging present around at least the LLM calls |
| 8 | **Documentation quality** | — | 5 | RESULTS.md clearly written, includes the required failure-case analysis, screenshots present and match the checklist |
| 9 | **Repo hygiene & reproducibility** | 7.3 | — (gate, not points) | `git clone` + `pip install -r requirements.txt` + running the scripts in order actually works for someone else. **A submission that doesn't run cannot score above 60/100 regardless of code quality**, since reproducibility was covered explicitly in Module 7. |

**Total: 100 points (+10 possible stretch points)**

### Scoring bands
- **90–100+**: Full pipeline + evaluation + at least one stretch goal, clean docs
- **75–89**: Full required pipeline works end-to-end, evaluation present, solid docs
- **60–74**: Pipeline mostly works, one or two required stages weak/incomplete but documented honestly
- **Below 60**: Pipeline doesn't run for the grader, or evaluation stage is missing/fabricated, or documentation doesn't reflect what was actually built

### What graders will actually do
1. Clone your repo fresh
2. Run `pip install -r requirements.txt`
3. Ask you (or check RESULTS.md) which provider/scenario to configure
4. Run `01` → `06` in order
5. Compare console output + your evaluation numbers against your RESULTS.md and
   screenshots
6. Spot-check one of your documented failure cases

**Honesty about limitations scores better than silence.** If reranking didn't work, if
RAGAS threw a dependency error, if your agent's HITL gate is the console `input()`
version instead of a real LangGraph interrupt — say so in RESULTS.md. That's exactly
the kind of engineering judgment this rubric is trying to measure.
