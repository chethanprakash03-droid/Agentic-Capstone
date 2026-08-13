# Setup Guide

Estimated time: **10 minutes**. Do this first, before touching any capstone logic.

---

## 1. Python environment

Requires Python 3.10+.

```bash
python --version          # confirm 3.10+

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If `pip install` is slow, that's normal the first time (sentence-transformers pulls a
model download on first use, not at install time).

---

## 2. Configure your model/API

Copy the example env file:

```bash
cp .env.example .env
```

Open `.env` and fill in **one** of the following blocks depending on what you have
access to. Leave the others blank/commented — `config.py` auto-detects which provider
is configured.

### Option A — Azure OpenAI (if your class quota still works)
```
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

### Option B — OpenAI API
```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

### Option C — Anthropic Claude API
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-haiku-4-5
```

### Option D — Any other OpenAI-compatible API (Groq, Mistral, Together, etc.)
```
LLM_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_BASE_URL=https://api.groq.com/openai/v1
OPENAI_COMPATIBLE_API_KEY=your_key_here
OPENAI_COMPATIBLE_MODEL=llama-3.1-8b-instant
```

### Option E — Ollama (fully local, free, no API key)
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```
Install Ollama first: https://ollama.com/download, then:
```bash
ollama pull llama3.2:3b
ollama serve   # if not already running
```

---

## 3. Sanity check your setup

Run:

```bash
python starter_code/config.py
```

You should see:
```
[OK] Provider detected: openai   (or whichever you configured)
[OK] Test call succeeded: "Hello from <model name>"
[OK] Embedding model loaded: all-MiniLM-L6-v2
```

If this fails, check:
- Is your `.env` in the repo root (same folder as `requirements.txt`)?
- Did you `pip install -r requirements.txt` inside the activated venv?
- For Ollama: is `ollama serve` running in another terminal?
- For Azure: has your quota actually expired? (common failure — switch to Option B/C/D/E)

---

## 4. Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | venv not activated | re-run `source venv/bin/activate` |
| `RateLimitError` / `429` | free tier limits | add short `time.sleep(1)` between calls, or switch provider |
| Embeddings very slow first run | model downloading | wait once, cached after |
| `ChromaDB` telemetry warnings | harmless | ignore, or set `ANONYMIZED_TELEMETRY=False` in `.env` |
| Ollama connection refused | server not running | run `ollama serve` in a separate terminal |
| RAGAS errors about event loop (Jupyter) | known RAGAS/Jupyter quirk | run evaluation as a `.py` script, not notebook |

Once the sanity check passes, move to `README.md` Section 5 (Stage 1).
