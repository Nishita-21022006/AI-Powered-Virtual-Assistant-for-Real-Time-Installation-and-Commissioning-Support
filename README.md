# AI-Powered Virtual Assistant — Installation & Commissioning Support

A retrieval-augmented assistant that answers system integrators'
installation and commissioning questions from product documentation, with
citations, and refuses when the documentation does not support an answer.

**Status: scaffold only.** Directories and module docstrings exist; no
implementation yet. Phase 0 (evaluation set) has not started.

- Architecture spec: `Assistant_Architecture.pdf`
- Working context for AI sessions: `CLAUDE.md`
- Source corpus: `Documents/`

## Why it refuses

The assistant answers only from passages retrieved at query time, and
shows a source for every claim. Below a retrieval score floor it stops
before generating and shows the relevant sections instead. An answer
with no citation is never displayed. In this domain one confidently
wrong wiring answer ends the project, so the target is ~85% of in-scope
questions answered at ~99% accuracy — not maximum coverage.

## Layout

```
app/
  main.py            FastAPI entrypoint — API gateway, stateless
  core/              config (model backend selection), logging
  api/routes/        query, ingest, jobs, feedback, health
  query/             10-stage synchronous pipeline, ~3s budget
  ingestion/         async pipeline, separate worker pool
  models/            generator | embedder | reranker | transcriber
                     one interface each, implementations behind it
  storage/           Postgres + pgvector, object storage
  cache/             semantic answer cache, invalidated on index change
  eval/              the evaluation harness — phase 0, run on every change
  observability/     full per-query tracing
tests/
  unit/              chunker warning rule, verification, score gate
  integration/       real Postgres; ingestion edge cases, retrieval, tables
  eval_set/          the 80–120 question evaluation set (phase 0)
scripts/             local ingest, re-embed, run eval
alembic/             migrations (first one enables pgvector)
```

The two pipelines are deliberately separate: ingestion is slow and heavy
(parsing, OCR, embedding), a query is fast and light. Sharing
infrastructure means an upload spike degrades live queries at exactly the
moment the product is supposed to be real-time.

## Setup in VS Code

Requires **Python 3.11+** and Git. No GPU, no Docker, no database.

**1. Clone it.** `Ctrl+Shift+P` → `Git: Clone` → paste the repository URL →
choose a folder → **Open** when prompted.

**2. Open the terminal** with ``Ctrl+` `` and run:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

python scripts/setup.py
```

When VS Code asks *"We noticed a new virtual environment, select it for
the workspace?"* — say yes.

The first run takes 10–20 minutes: it installs PyTorch and downloads
~2.5 GB of models. It prints each step, and is safe to re-run if the
connection drops — completed steps are skipped.

**3. Start it:**

```bash
python -m uvicorn app.main:app --port 8000
```

Wait for `Application startup complete` — about a minute while the
models load into memory — then Ctrl+click the
<http://127.0.0.1:8000> link in the terminal.

Check `python --version` first if setup complains; 3.10 and earlier
will not work.

## What setup.py does

Installs dependencies, downloads the embedder and reranker (~2.5 GB,
once), builds the search index from the PDFs in `Documents/`, and
smoke-tests retrieval end to end. Every step checks whether it is
already done, so re-running after an interrupted download resumes
rather than starting over.

It fetches only the weight format actually used. Left to itself the
library pulls safetensors, PyTorch and ONNX copies of the same model,
which is roughly 9 GB instead of 2.5.

## The API key

Everything except the written answer runs locally and free. Only the
generator calls an API.

Without a key the assistant still works — it finds and shows the right
sections, it just doesn't write prose. That is the architecture's
degraded mode, not a broken state.

For written answers, get a free key at <https://console.groq.com>
(no card) and put it in `.env`:

```
GROQ_API_KEY=your-key-here
GENERATOR_MODEL=qwen/qwen3.8-27b
```

Free tier is roughly 30 requests/minute and 1,000/day — ample for
development, not for a hundred concurrent users.

### Postgres (optional, not yet required)

The index is held in memory and cached to `.cache/`, which is sufficient
for this corpus. The pgvector schema and migration exist for when
persistence is needed:

```bash
docker compose up -d
alembic upgrade head
```

### A note on the pinned versions

The pins in `pyproject.toml` were verified against the live PyPI index
with `pip index versions` on 2026-10-10, and the set resolves cleanly
under `pip install --dry-run`. That check ran under Python 3.13 — every
package declares a `requires-python` floor at or below 3.11, but re-run
the dry-run in a 3.11 venv before committing a lockfile if 3.11 is the
real target.

Three of them are a major line past what most tutorials assume, so copied
config may not apply: **sqlalchemy 2.1.x** (past the long-lived 2.0),
**mypy 2.x**, and **openai 3.x**. Note also that `openai` 3.x depends on
`httpx2`, a separate distribution from `httpx` — they coexist rather than
conflict, but pinning both puts two HTTP stacks in the environment.

## Model backends

Four roles sit behind interfaces in `app/models/`. Switching any of them
is a change to `.env`, never to calling code.

| Role | Now — cloud | Later — local |
|---|---|---|
| Generator | A hosted frontier model via API (Claude, GPT or Gemini) | Qwen 3 8B, 4-bit quantized |
| Embedder | Gemini Embedding 001, or a retrieval-focused option such as Voyage | BGE-M3 |
| Reranker | Cohere Rerank or a hosted cross-encoder | BGE reranker |
| Transcriber | Deepgram or AssemblyAI | NVIDIA Parakeet TDT 0.6B v2, faster-whisper fallback |

This is section 6 of the PDF as written. Cloud first keeps model hosting
off the critical path — GPUs are slow to provision and expensive to get
wrong, and a strong cloud model means a wrong answer points at retrieval
rather than at a weak generator hiding our mistakes. Everything speaks an
OpenAI-compatible interface where the provider allows it, which is what
keeps the Ollama (development) to vLLM (production) to hosted-API moves
configuration.

The generator's frontier vendor is not yet picked — the PDF leaves it
open. Choose one before phase 3 and add its SDK then. Cloud and local
embedding vectors are incompatible, so extracted text and chunk
boundaries are stored separately from the vectors from day one: changing
the embedder is a batch re-embed, not a re-parse.

## Build order

Phase 0 evaluation set → 1 retrieval only → 2 reranker (measured) →
3 generation + verification → 4 web UI. Phases 0–4 are the MVP; voice,
local profile and mobile follow.

Do not skip phase 0. Without known-answer questions, no later change can
be shown to have helped.
