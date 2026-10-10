# AI-Powered Virtual Assistant — Installation & Commissioning RAG

This file is the working context for future sessions. The full spec is
`Assistant_Architecture.pdf` (v1 draft, reviewed 2026-10-09/10) — read that PDF
directly for anything not captured here. `Documents/` holds the real source
corpus for ingestion (product manuals, an installation-guide family of PDFs,
and a product catalog) — treat it as input data, not reference material.

## What we're building

A retrieval-augmented assistant for system integrators installing and
commissioning equipment on site. It answers from product documentation —
manuals, installation guides, wiring diagrams, FAQs, past issue resolutions —
with text and voice, on web and mobile, online or offline. Target shape:
answers ~85% of in-scope questions, ~99% accurate on those, and says
"I don't know, here are the relevant pages" on the rest. Abstention is a
feature, not a failure.

The domain is narrow and predictable (wiring, fault codes, step sequences,
parameter settings, part compatibility) — not "ask anything." Sales,
warranty, and general troubleshooting questions are legitimately out of
scope and can be refused.

This is mostly a retrieval problem, not a model problem: if search returns
the wrong passage, no generator can produce a correct answer; if it returns
the right passage, even a small model can read it correctly. Most engineering
effort goes into retrieval quality.

## Non-negotiables

- **No answers from model knowledge.** The assistant answers only from text
  retrieved at query time and shown as a citation. No citation → the answer
  is not shown.
- **Abstention is a feature.** Below a retrieval score floor, the system
  refuses and shows the relevant sections instead of generating. This is a
  deterministic threshold check, not a model judgment call.
- **Safety warnings travel with their procedure.** A chunk that contains a
  step contains that step's warnings. Never split a warning from the
  procedure it governs — enforced in the chunker, tested for.
- **Ingestion and query serving are separate infrastructure.** Ingestion is
  async, on its own worker pool; a query is synchronous and latency-budgeted
  (~3s to first words). An upload spike must never degrade a live query.
- **Generator, Embedder, Reranker, Transcriber each sit behind one interface**
  with an API-served implementation now and a local implementation later.
  Swapping is a config change, not a code change. All API-served
  implementations speak an OpenAI-compatible interface.
- **Raw text and chunk boundaries are stored separately from vectors.**
  Changing the embedding model is a re-embed batch job, not a re-parse of
  every document.

## Model selection (PDF section 6 — ALL SLM)

Four roles, all small open-weight models. **Run it locally unless there
is a reason not to.** Three of the four run on an ordinary laptop CPU for
free; only the generator is served through an API, because it is the one
role where CPU speed genuinely hurts — a 4B–8B model on CPU produces a
few tokens per second, which breaks the three-second target.

| Role | Model | Where it runs | Size |
|---|---|---|---|
| Embedder | BGE-M3 | Local, CPU — free | ~2 GB |
| Reranker | BGE reranker (m3 family) | Local, CPU — free | ~1–2 GB |
| Transcriber | NVIDIA Parakeet TDT 0.6B v2 (faster-whisper fallback) | Local, CPU — free | ~0.7 GB |
| Generator | Qwen 3 via OpenRouter `:free` (self-hosted 4B/8B later) | API — free tier | 3–6 GB if self-hosted |

BGE-M3 does dense *and* sparse retrieval in one model, which is exactly
the hybrid search this system needs without running two. Parakeet is
open-weight, so no NVIDIA API account is needed.

**Cost: zero, or ~$10 for convenience.** OpenRouter's free tier is
20 requests/minute and 50 requests/day; buying ≥10 credits once raises
the free-model allowance to 1,000/day, which is recommended because
50/day is restrictive while developing.

Two things to watch on the free tier: a **negative OpenRouter balance
returns 402 even on free models**, so keep it at or above zero; and
20 req/min is fine for two developers but nowhere near a
hundred-concurrent-user demo — that needs paid inference or our own
serving. Hugging Face's Inference API is not an alternative (free
accounts get no monthly credits).

**Hardware: no GPU for phases 0–6.** RAM is what matters — ~8 GB free
runs the embedder and reranker, 16 GB is comfortable with voice too. A
GPU is only needed at phase 7, when the generator moves in-house.

Two consequences worth keeping in mind. Documents never leave our
machines during phases 0–4, since only the generator is remote and it
only ever sees a handful of retrieved passages. And because every model
is open-weight, offline deployment is the same models served locally —
a config change, not a rewrite.

The expensive-to-reverse decision: changing embedding model means
re-embedding the whole corpus. Extracted text and chunk boundaries are
stored separately from the vectors from day one, making that a batch
re-embed rather than a re-parse of every document.

## Build order (phases 0–4 = MVP)

0. **Evaluation set** — 80–120 real questions with known answers. Nothing
   else is measurable without this. Do not skip.
1. **Retrieval only** — ingestion + hybrid (meaning + exact-word) search, no
   chat interface, no generation. Prove we can find the right passage.
2. **Reranker** — add it, measure the before/after delta on the eval set.
   Don't assume it helps; show it.
3. **Generation + verification** — answer generation with citations, plus
   guardrail tiers 1–3 (retrieval score floor, prompt-level constraints,
   deterministic output verification: every number/part code must appear
   verbatim in a cited source; cited document must be in the retrieved set).
4. **Web chat UI** — demoable. This is the MVP.

Later phases (not MVP): 5 async upload pipeline hardening, 6 voice input
(push-to-talk), 7 self-host the generator — the only role still on an
API, and the one thing that needs a GPU; everything else is already
local — 8 mobile app.

Degraded mode / fallback chain: generator API → self-hosted model →
retrieval-only passages. Because the embedder and reranker are local,
retrieval keeps working with no connectivity at all; only the written
answer is lost. The system never returns nothing.

Build the expensive-to-retrofit boundaries early even in trivial form —
one worker instead of a pool, one model instead of a chain, tiers 1 and 3
only — but build them. Don't build the complete fallback topology before
there's a working demo answering real questions correctly.

## Answered

**PDF open question 2 — approval to send data to a third-party model
API: YES, approved (2026-10-10).** The corpus in `Documents/` is test
material supplied for exactly this purpose. Note the all-SLM design has
also made this much narrower than it was: the embedder, reranker and
transcriber are local, so whole documents never leave the machine — only
the handful of retrieved passages sent to the generator do. Do not
re-raise this.

**Corpus redistribution — permitted (2026-10-10).** The manuals in
`Documents/` carry a Honeywell copyright notice forbidding redistribution,
but Honeywell have explicitly permitted their use for this project. The
repository is PRIVATE and the PDFs are committed to it deliberately. Do
not re-raise this, and do not make the repository public without
re-checking that permission.

## Open questions (still unanswered)

Numbered as in PDF section 9, so Q2 is absent — it is answered above.

- **Q1** — "100 users at once": 100 concurrent requests, or 100 registered
  users? Decides whether real GPU serving and batching are needed.
- **Q3** — What hardware will on-premise sites actually have? VRAM is the
  hard constraint and it picks the local model size: 4B, 8B or 14B.
- **Q4** — Full corpus size and revision frequency? Decides whether
  Postgres + pgvector is sufficient or a dedicated vector database is
  needed, and how much the version manager matters.
- **Q5** — How does an offline/air-gapped site receive updated
  documentation? Unspecified in every version of the architecture; needs
  a defined mechanism even if the answer is "a bundle on a USB drive".
- **Q6** — Should all integrators see all documents? If not, access
  control becomes metadata filtering — far easier to design in now
  (app/query/filters.py) than to retrofit.
- **Q7** — Which model licences are acceptable for commercial use? "Open
  weights" is not "free for commercial use". Confirm per model for the
  local profile (Qwen 3, BGE-M3, Parakeet) before building on one.

Blocking phase 1: the phase 0 evaluation set must be written first, and
it needs someone who knows the equipment to verify the answers.
Blocking phase 3: pick the generator's frontier vendor.
