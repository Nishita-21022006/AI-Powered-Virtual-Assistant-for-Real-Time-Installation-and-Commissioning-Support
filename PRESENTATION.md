# AI-Powered Virtual Assistant for Real-Time Installation and Commissioning Support
## Presentation content — slide by slide

> **How to use this file.** Each `## Slide N` is one slide. The bullets are
> what goes *on* the slide — keep them short on the actual slide. The
> **Speaker notes** are what you say, not what you show. **Do not paste the
> speaker notes onto the slide.**
>
> Every number here was measured on 2026-10-11 and is reproducible from the
> repository. Where something is unmeasured or uncertain, it says so. Do not
> round the numbers up when presenting — the honesty is the strongest part
> of this project.

---

## Slide 1 — Title

**AI-Powered Virtual Assistant for Real-Time Installation and Commissioning Support**

A retrieval-based assistant that answers system integrators' questions
from product documentation — with a citation for every claim, and a
refusal when the documentation does not cover it.

- Built on 7 Honeywell / Ackermann-Clino nurse-call installation manuals
- 274 searchable passages · 104-question evaluation set
- Runs on a laptop. No GPU. Total running cost: zero.

**Speaker notes:** Open with the problem, not the technology. The audience
does not care that it is RAG yet.

---

## Slide 2 — The problem

An integrator is on site, halfway through installing a unit, and has a
question.

Today they either:
- stop work and search a 40-page PDF, or
- call technical support and wait

Both cost time on a job that is already halted.

**Speaker notes:** This is the whole motivation. Everything after this is
detail about how to answer that question reliably enough to be trusted.

---

## Slide 3 — Three words that shaped every decision

| Word | What it forces |
|---|---|
| **Real-time** | Latency is a feature. Target: answer on screen in under 3 seconds. |
| **Installation and commissioning** | A narrow, predictable domain — wiring, fault codes, step sequences, parameters, compatibility. Not "ask anything". |
| **Support** | Not a replacement for tech support. Handing a hard question to a human is the system working. |

**Speaker notes:** The third one is the key to the whole design. Because we
are *supporting* rather than replacing, refusing is allowed — and that
permission is what makes the accuracy target reachable.

---

## Slide 4 — What success looks like

Not "it answers everything."

- Answers roughly **85%** of in-scope questions
- Is roughly **99% accurate** on the ones it answers
- Says **"I don't know — here are the relevant pages"** on the rest

> An assistant that answers everything at 90% accuracy sounds better and is
> worse. One confidently wrong wiring answer ends the project.

**Speaker notes:** Say the last line slowly. This is the thesis of the whole
project and the justification for everything that follows.

---

## Slide 5 — The core design principle

**The assistant never answers from the model's own knowledge.**

It answers only from text it has just retrieved from the documentation, and
it shows the source for every claim. If it cannot find the source, it does
not answer.

**Speaker notes:** This is a hard constraint, not a preference. It is
enforced in code — an answer with no citation is discarded before the user
ever sees it.

---

## Slide 6 — Why not just train a model on the manuals?

Three failures, any one of which is fatal here:

1. **It cannot cite sources** — nobody can verify an answer
2. **It goes stale** the moment a manual is revised
3. **Models recalling facts from training produce plausible wrong ones** —
   exactly the failure we cannot afford

Instead: keep the documentation searchable, retrieve the right paragraphs,
and ask the model to *read* them.

**Speaker notes:** The model is doing reading comprehension, not recall.
That distinction is the entire architecture in one sentence.

---

## Slide 7 — How it works, in six lines

**Question:** "What's the maximum screw head height when wall-mounting?"

1. **Search** the documentation for passages about wall mounting and screws
2. **Find** three candidates; the best is §2.2 "Installation on a wall", p.8
3. **Paste** that passage into the model's prompt with the question
4. **Model reads** and writes: "Maximum 3 mm. Exceeding it risks damaging
   the electronics."
5. **Check** that "3 mm" actually appears in the passage — it does
6. **Show** the answer with its citation

**Speaker notes:** Walk through this slowly. If they understand this slide
they understand the product. Everything else is doing these six steps
reliably at scale.

---

## Slide 8 — This is a retrieval problem, not an AI problem

Steps 1 and 2 decide whether the answer is right.

- If search returns the **wrong** passage, no model — however good — can
  produce a correct answer
- If search returns the **right** passage, even a small model reads it
  correctly

**So most engineering effort went into retrieval quality, not the language
model.**

**Speaker notes:** Teams that get this backwards spend weeks comparing
models and ship something that cannot find the right page. We built and
measured retrieval first, with no chat interface at all.

---

## Slide 9 — System architecture (7 layers)

```
   Web chatbot                    Mobile app (later)
        |                               |
        +---------------+---------------+
                        |
                 API gateway
          auth · rate limits · sessions · logging
                        |
          +-------------+-------------+
          |                           |
   Query pipeline              Ingestion pipeline
   synchronous                 asynchronous
   latency-budgeted            own workers
          |                           |
          +-------------+-------------+
                        |
                    Storage
      Postgres: vectors + keyword index + metadata
      Object storage: original files + figure images
                        |
                  Model layer
    Generator | Embedder | Reranker | Transcriber
        (one interface, swappable implementations)
                        |
                  Observability
   every query logged with passages, scores, latency, verdict
```

**Speaker notes:** The important thing is not what is in each box, but that
the boundaries are real. Each boundary is where we can swap a cloud service
for a local one without rewriting anything above or below it.

---

## Slide 10 — Why ingestion and querying are separated

**The single most important scalability decision.**

- **Ingestion is slow and heavy** — parsing PDFs, OCR, generating embeddings
- **A query is fast and light**

If they share a machine, the moment someone uploads a batch of documents
*every integrator on site sees their answers slow to a crawl* — precisely
when the product is supposed to be real-time.

→ Uploads go on a queue and return immediately. A separate worker pool
drains it. An upload spike can never degrade a live query.

**Speaker notes:** This is the kind of boundary that is cheap to build early
and expensive to retrofit, which is why it went in before it was needed.

---

## Slide 11 — The query path (10 stages)

| # | Stage | Budget | What it does |
|---|---|---|---|
| 1 | Normalize | ~0ms | Trim, reject empty or absurd input |
| 2 | Semantic cache | ~20ms | Hit → return instantly |
| 3 | Rewrite with history | ~150ms | "and the other one?" → standalone question |
| 4 | Metadata filter | ~0ms | This product, current revisions only |
| 5 | **Hybrid retrieve** | ~200ms | Meaning search + exact-word search → ~50 candidates |
| 6 | **Rerank** | ~250ms | Cross-encoder re-sorts → top 5 |
| 7 | **Score gate** | ~0ms | Too low? Stop. Never generate from weak evidence |
| 8 | Generate | ~1200ms | Answer from those 5 passages, with citations |
| 9 | **Verify** | ~50ms | Every figure must appear in a source. Fails? Reject |
| 10 | Respond | — | Answer + document, section, page per claim |

Stages **7** and **9** are the two places the system is allowed to refuse.

**Speaker notes:** Measured end-to-end latency is about 2.9 seconds —
retrieve 192ms, rerank 2.2s, generate 568ms, verify 2ms. Inside the budget,
with reranking dominating.

---

## Slide 12 — Hybrid retrieval: why two searches

Technical documentation is full of **exact strings**: part numbers, fault
codes, board revisions, terminal labels.

- **Meaning-based search** understands "the mounting screw" but treats
  `72591D` as noise
- **Exact-word search** finds `72591D` but misses "how do I fix it to the
  wall"

→ Run both, merge the rankings. Dense vector search (BGE-M3) + BM25 keyword
search, combined with Reciprocal Rank Fusion.

**Speaker notes:** RRF combines *rankings* rather than scores, because a
cosine similarity and a BM25 score are not the same quantity — averaging
them would invent a comparison the numbers do not support.

---

## Slide 13 — The ingestion path

```
Upload → Hash check → Queue → Workers
         (seen it?)   (absorbs  (own pool)
                       spikes)
                                   |
         +-------------------------+
         |
   Parse → Chunk → Embed → Tag
   text,   by       into    product,
   tables, section  vectors version,
   figures                  page
         |
   Version manager — retire old revision's chunks
         |
   Index + invalidate the cache
```

**Edge cases designed for:** duplicate upload, new revision, scanned
image-only PDF, corrupt/password-protected file, oversized file, worker
crash mid-job, two of three files failing, embedding model swap.

**Speaker notes:** Jobs are idempotent and retryable — a worker dying
mid-job never leaves a half-indexed document visible.

---

## Slide 14 — The decision that matters most: chunking

A "chunk" is one searchable unit of text. **Chunking badly is the most
common way to build a RAG system that retrieves correctly and still gives
dangerous answers.**

**✗ Naive — split every 500 words**
```
--- chunk 41 ---
...Step 3: Align the panel.
Step 4: Tighten the four mounting
screws and fit the terminal.

--- chunk 42 ---
WARNING: Maximum screw head height
is 3 mm, otherwise the electronics
may be damaged.
```

Someone asks "how do I mount it?" → chunk 41 is retrieved. It is the
**correct** passage. The answer is complete, confident — and silently omits
the warning that stops them destroying the unit.

**Speaker notes:** This is the most important slide in the deck. Pause here.

---

## Slide 15 — The hard rule

> **A safety warning is never separated from the procedure it governs.**
> If a chunk contains a step, it contains that step's warnings.

- Enforced in the chunker
- Tested for — including against the real manuals, not just fixtures
- When a section must be split, **all** of its warnings are attached to
  every piece containing steps

**Speaker notes:** We duplicate rather than guess. The document does not
record which warning governs which step, so inferring it means guessing.
Duplication costs a little index space and makes the guarantee
unconditional. Given the asymmetry — a wasted chunk versus someone
destroying a unit — that is the right trade.

---

## Slide 16 — Finding #1: the warnings are pictograms

**We expected to find "WARNING:" in the text. There is no such string
anywhere in this corpus.**

The manuals define warnings on a legend page as **images** — a triangle
pictogram drawn in the left margin beside indented text.

- A keyword-based chunker would have found **zero** warnings
- It would have reported every chunk safe
- The system's core guarantee would have been silently broken, with all
  tests passing

→ Warning detection is **layout-based**: triangle vector glyph, sized
18–48pt, positioned left of text and vertically overlapping its first line.

**Result: 10 warnings and 9 notes correctly detected in one manual alone.**

**Speaker notes:** This was found by an agent reading the PDFs closely to
write evaluation questions, before the chunker was finished. Had we not
looked, we would have shipped something that looked correct and was not.

---

## Slide 17 — Model selection: all small, all open-weight

| Role | Model | Where | Size | Cost |
|---|---|---|---|---|
| **Embedder** | BGE-M3 | Local, CPU | ~2 GB | free |
| **Reranker** | ms-marco-MiniLM-L6-v2 | Local, CPU | 88 MB | free |
| **Transcriber** | Parakeet TDT 0.6B v2 | Local, CPU | ~0.7 GB | free |
| **Generator** | Qwen3.8-27B via Groq | API | — | free tier |

**Principle: run it locally unless there is a reason not to.**

Only the generator is remote, because a 4B–8B model on CPU produces a few
tokens per second — which breaks the three-second target.

**Total cost of the MVP: zero.** No GPU required for phases 0–6.

**Speaker notes:** Three of four roles need no account at all. Without an
API key the assistant still works — it finds and shows the right sections,
it just does not write prose. That is designed degraded mode, not failure.

---

## Slide 18 — Finding #2: the specified reranker is unusable

The architecture specified **BGE-reranker-v2-m3** and budgeted 250ms for
the rerank stage.

| Reranker | Params | Per query | vs. budget |
|---|---|---|---|
| bge-reranker-v2-m3 *(specified)* | 568M | **42.3 s** | 170× over |
| bge-reranker-base | 278M | 11.1 s | 44× over |
| **ms-marco-MiniLM-L6-v2** | **22M** | **2.1 s** | 8× over |

**The 22M model beat the 278M model on every accuracy metric** *and* ran
5× faster. 12× more parameters bought nothing.

**Speaker notes:** The spec said this reranker "runs fine on modest
hardware". That claim does not hold on CPU at that model size. We only know
because we measured instead of assuming. Honest caveat: MiniLM is
English-only, and the corpus contains untranslated German, which our
English-only evaluation set structurally cannot detect.

---

## Slide 19 — Safety guardrails: six tiers

| Tier | What | Mechanism |
|---|---|---|
| **0 · Input** | Reject nonsense; show voice transcripts back before acting | code |
| **1 · Retrieval** | Score floor — below it, abstain without generating | code |
| **2 · Generation** | Answer only from supplied passages; never paraphrase a number | prompt |
| **3 · Output** | Every figure must appear in a cited source; no citation → never shown | code |
| **4 · Escalation** | Say confidence is low; show closest sections; name the document; offer support | code |
| **5 · Always on** | Visible citations; standing "supplements, not replaces" line; feedback | UI |

**Tiers 1 and 3 are plain code, not AI — they cannot themselves
hallucinate.**

**Speaker notes:** Note how little of this is machine learning. The checks
that matter most are string comparisons: cheap, testable, and incapable of
inventing anything.

---

## Slide 20 — Build plan: 9 phases, 0–4 are the MVP

| # | Deliverable | What it proves | Status |
|---|---|---|---|
| 0 | Evaluation set — 104 questions | That we can measure anything at all | ✅ |
| 1 | Ingestion + hybrid retrieval | That we can find the right passage | ✅ |
| 2 | Add the reranker, measure the delta | That reranking earns its latency | ✅ |
| 3 | Generation + citations + tiers 1–3 | That answers are trustworthy | ✅ |
| 4 | Web chat interface | Demoable. This is the MVP | ✅ |
| 5 | Async upload pipeline | Concurrent uploads don't degrade queries | ⬜ |
| 6 | Voice input (push-to-talk) | Hands-free use on site | ⬜ |
| 7 | Self-host the generator | The offline / on-premise requirement | ⬜ |
| 8 | Mobile app | Full scope | ⬜ |

**Speaker notes:** Phase 0 first, always. Every team building this skips the
evaluation set and then spends weeks arguing about whether a change helped.

---

## Slide 21 — Phase 0: the evaluation set

**104 questions with verified answers.** Every supporting quote
machine-checked verbatim against the PDFs.

**70 answerable**
| Category | Count |
|---|---|
| Parameter | 19 |
| Wiring | 16 |
| Step sequence | 13 |
| Compatibility | 12 |
| Installation | 7 |
| Fault code | 3 |

11 carry safety warnings · 22 come from tables

**34 abstention** — questions it must refuse
out-of-domain 6 · not-in-corpus 6 · ambiguous 6 · needs-missing-document 6 ·
version-dependent 5 · unsafe-to-guess 5

**Speaker notes:** A set containing only answerable questions measures
coverage and cannot measure the thing that matters most — whether it refuses
when it should. The abstention half was deliberately written to look
*plausible*, not absurd, because the hard case is the question that looks
answerable and is not.

---

## Slide 22 — Phase 1 results: retrieval

7 manuals → 230 pages → **274 passages** → BGE-M3 embeddings, local CPU

| Metric | Raw | Against ceiling |
|---|---|---|
| recall@1 | 0.357 | 0.391 |
| recall@5 | 0.686 | **0.750** |
| recall@10 | 0.757 | 0.828 |
| recall@50 | 0.857 | **0.938** |
| MRR | 0.495 | — |

70 queries in 21 seconds.

**"Against ceiling" is the number to trust:** 6 of 70 eval quotes cannot be
matched against our parser's text at all — a different PDF tool wrote them
— so **0.914 is the hard maximum**, not 1.0.

**Speaker notes:** Measuring against a fictional 1.0 would mean chasing six
questions that are unreachable by construction. Being explicit about your
ceiling is part of measuring honestly.

---

## Slide 23 — Phase 2 results: does reranking earn its cost?

**The shape of the phase 1 result set up phase 2 perfectly:**
- Right passage in the top-50: **0.938**
- Right passage in the top-5: **0.750**

That ~19-point gap is exactly what a reranker exists to close.

| | recall@1 | recall@5 | MRR |
|---|---|---|---|
| Retrieval only | 0.357 | 0.686 | 0.495 |
| **+ reranking** | **0.443** | **0.729** | **0.559** |
| Delta | **+8.6 pts** | +4.3 pts | +0.064 |

Reranking captures **83%** of the reachable headroom.

**Speaker notes:** The architecture claimed reranking was "the biggest
single accuracy win in the pipeline". We now have a measured before-and-
after instead of an assertion. The gains are largest at the sharp end —
rank 1 improved most.

---

## Slide 24 — Phase 3 results: end to end

All 104 questions through the complete pipeline.

| Metric | Value |
|---|---|
| **Answer rate** | **0.829** (58/70) — target ~85% ✅ |
| **Accuracy** (contradictions only) | **0.862** |
| Accuracy (strict judge) | 0.603 |
| Abstention precision | 0.588 (20/34) |
| Cited the expected document | 55/58 (95%) |

**Why two accuracy numbers?** The automated judge marks an answer wrong if
*any* expected figure is missing — which conflates **omission** with
**error**. Of 23 answers it flagged, **8 were genuine contradictions and 15
were omissions**.

**Speaker notes:** Report 0.862 and say explicitly which figure it is and
why. Do not quote 0.603 as the accuracy — it understates — and do not quote
0.862 without the explanation. The auditing script that separates them is
in the repo.

---

## Slide 25 — Finding #3: all 8 real errors had one cause

| Asked about | Answered from | Consequence |
|---|---|---|
| SEM temperature range (+5…+55°C) | **SEM-IP** → +5…+40°C | wrong variant |
| Bed power terminals (max 2 A) | ward power supply → **10 A** | **5× over the limit** |
| R-ABUS (6 devices, 200 m) | **ZBUS** → 127 devices, 2500 m | wrong bus |

**Every one is the right answer to a different question.**

Not fabrication — each figure is real and correctly cited. The system
retrieved a plausible neighbour and answered confidently from it.

**Speaker notes:** The bed-terminal one is the dangerous example. An
installer told 10 A when the limit is 2 A could draw five times the rated
current.

---

## Slide 26 — Finding #4: verification checks provenance, not relevance

**Tier 3 passed every one of those 8 errors — and it was right to.**

`"max. 10A"` *does* appear verbatim in a cited source. The verifier
confirms a figure came from a retrieved document. It cannot confirm the
figure answers the question.

> The architecture treats tier 3 as the net that catches "a confident
> invented value". It does catch those. It does not catch **correctly-quoted
> wrong values** — which on this evidence are the more common dangerous
> failure.

**Speaker notes:** This is the most valuable thing the evaluation found —
a class of failure nobody had articulated, including the architecture
document. Worth presenting as the headline result.

---

## Slide 27 — Finding #5: the score floor cannot carry abstention

The architecture makes the retrieval score floor tier 1 — the deterministic
gate that refuses before generating.

**Calibrated against the evaluation set:**

| | median | range |
|---|---|---|
| Answerable questions | 2.17 | −7.29 … 8.34 |
| Out-of-scope questions | 1.62 | −7.22 … 7.13 |

**The distributions overlap almost completely. No threshold separates
them.**

At −5 you keep 96% of real questions but catch only 12% of out-of-scope
ones. At +2 you catch 56% of out-of-scope but throw away half your real
answers.

→ The floor is now a loose backstop. **Abstention is carried by tier 2 (the
model declining) and tier 3 (verification)** — both of which demonstrably
work.

**Speaker notes:** This happens because the abstention questions were
deliberately written to look plausible. A refusal set of obviously-absurd
questions would have "passed" and taught us nothing.

---

## Slide 28 — The fix: entity-aware answering

All 8 errors were confusable **near-synonyms**, and the names are substrings
of one another: `SEM` inside `SEM-IP`, `ABUS` inside `R-ABUS`.

**Entities grouped into families of mutually exclusive alternatives:**
- bus: R-ABUS · DBUS · ZBUS · ABUS · BBUS
- room module: SEM-IP · SEM+ · SEM
- unit: SCVIP · SCIP+ · STIP · SINF · SCL · SRT · SRD · SBU

Within a family, naming one **rules out** the others — so a passage about a
sibling is **evidence against** relevance, not merely neutral.

**Two changes:**
1. **Retrieval** demotes a sibling's passage (score adjustment, not a filter)
2. **The generator** names what its answer is about, and declines rather
   than answering from a neighbour

**Speaker notes:** The first attempt made it *ask* "which device?" for a
question it had previously answered well — correct and useless. The failure
was never answering an ambiguous question; it was answering *without saying
which thing the answer described*. Naming the subject fixes it without
stalling someone with a halted job.

---

## Slide 29 — Phase 4: the interface

- **Chat interface** with conversation history, persisted locally
- **Steps render as a tickable checklist** — state survives a reload, so a
  procedure is something you *work from*, not just read
- **Citations on every claim** — document · section · page
- **Safety warnings in a dedicated amber block**; amber is used for nothing
  else, so a warning can never be confused with emphasis
- **Refusals styled with conviction**, not as errors
- **Document upload** — add or supersede a manual, parsed and indexed live
- Monospace for part numbers; larger touch targets on mobile for gloved,
  one-handed use on site

**Speaker notes:** Demo the refusal, not just the answer. The refusal is the
product's claim.

---

## Slide 30 — Live demo script

Run in this order:

1. **"How do I mount the terminal on a wall?"**
   → numbered steps, each cited, warning in amber. Tick two steps, reload
   the page, show they persist.

2. **"What is the maximum screw head height?"**
   → the value stated immediately, with its warning carried automatically

3. **"Is the USB-C port functional?"**
   → "No, currently without function" — synthesised across **two**
   documents, both cited

4. **"What is the warranty period and price?"**
   → **refuses**. "I only answer installation and commissioning questions."

**Speaker notes:** #4 is the one to dwell on. Anything can answer; refusing
correctly is the hard part. If the first query comes back as sources only,
say so — the free tier allows ~3 questions a minute and recovers in about a
minute.

---

## Slide 31 — Engineering practices

- **118 automated tests**, all passing — ~5,100 lines of application code,
  ~1,100 lines of tests
- **Evaluation re-run on every significant change** — resumable, rate-limit
  aware, 104 questions in ~40 minutes
- **Full query tracing** — for any answer we can reconstruct exactly which
  passages produced it, their scores, and the verdict
- **Config-driven model selection** — switching provider or model is a
  change to `.env`, never to code
- **One-command setup** — `python scripts/setup.py` installs, downloads,
  indexes and smoke-tests

**Speaker notes:** The eval discipline is the part worth emphasising. Every
claim in this deck is reproducible by running one command in the repo.

---

## Slide 32 — Known limitations (stated deliberately)

- **Retrieval does not reliably surface the right product's manual.** For
  the SEM temperature question, all five passages came from the SEM-IP
  manual. Entity scoring demotes siblings but cannot promote a document that
  was never retrieved.
- **Wiring diagrams are not yet shown.** The parser extracts figure images;
  we do not yet store or serve them. For spatial questions this matters.
- **Accuracy is 0.862, against a target of 0.99.** Not there yet.
- **Some headings are swallowed by the table extractor**, so a few chunks
  still mix subjects.
- **The English-only reranker** cannot be assessed on the corpus's German
  sections by an English-only eval set.
- **Free-tier throughput is ~3 questions/minute** — fine for development,
  nowhere near a hundred concurrent users.

**Speaker notes:** Volunteer these before you are asked. A panel trusts a
project more when it knows its own weaknesses precisely.

---

## Slide 33 — What's next

| Priority | Work | Why |
|---|---|---|
| 1 | **Entity-aware retrieval**, completed | Turns honest declines into correct answers |
| 2 | **Inline wiring diagrams** | Spec calls this a known weak point; extraction already exists |
| 3 | **Phase 5** — async ingestion | Required before concurrent uploads |
| 4 | **Phase 6** — voice | Whisper already free on the current account |
| 5 | **Phase 7** — self-host generator | The offline requirement. Needs a GPU |

**Speaker notes:** Priority 1 is the one that moves the accuracy number.

---

## Slide 34 — Open questions for the customer

Decisions the architecture cannot make:

1. **"100 users at once"** — 100 concurrent requests, or 100 registered
   users? Changes the serving requirement enormously.
2. **What hardware will on-premise sites have?** VRAM picks the local model
   size.
3. **How large is the full corpus, and how often revised?** Decides whether
   Postgres is sufficient.
4. **How does an air-gapped site receive updated documentation?** Currently
   unspecified in every version of the design.
5. **Should all integrators see all documents?** If not, access control is
   metadata filtering — far easier to design in now than to retrofit.

**Speaker notes:** Also worth raising: the corpus is almost entirely
*installation* documents. The manuals cross-reference ~14 commissioning
documents we do not have. The product is called Installation **and
Commissioning** — half of that is not in the corpus.

---

## Slide 35 — Summary

**Built:** phases 0–4 of a 9-phase plan — a working assistant that answers
installation questions from real manuals, with citations, and refuses when
it should.

**Measured:** 0.829 answer rate · 0.862 accuracy · 274 passages · 118 tests

**Cost:** zero. Three of four models run on a laptop CPU.

**Learned:** five things that corrected the architecture document, each
found by measuring rather than assuming — most importantly that
**verification guarantees provenance, not relevance**.

> The evaluation set was the highest-value day of the project. Every number
> in this presentation exists because of it.

**Speaker notes:** End on the eval point. It is the transferable lesson.

---

## Appendix A — Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | |
| API | FastAPI + Uvicorn | |
| PDF parsing | pdfplumber + pypdf | Per-word geometry for layout-based warning detection. PyMuPDF rejected — faster but AGPL, and this is commercial |
| Embedding | BGE-M3 (local, CPU) | Dense *and* sparse in one model; multilingual |
| Lexical search | BM25 (in-process) | Exact part numbers and fault codes |
| Fusion | Reciprocal Rank Fusion | Combines rankings, not incomparable scores |
| Reranking | ms-marco-MiniLM-L6-v2 | Measured best accuracy *and* 20× faster than the alternatives |
| Generation | Qwen3.8-27B via Groq | Free tier; OpenAI-compatible so the provider is a config change |
| Storage | In-memory + disk cache; Postgres + pgvector schema ready | 274 chunks does not need a database yet |
| Frontend | Single-page HTML/CSS/JS | No build step |
| Testing | pytest — 118 tests | |

---

## Appendix B — Reproducing every number

```bash
python scripts/setup.py                  # install, download, index
python -m pytest tests/unit -q           # 118 tests
python -m app.eval.run                   # phase 1: retrieval
python -m app.eval.run --rerank          # phase 2: reranking delta
python -m app.eval.end_to_end            # phase 3: full pipeline (~40 min)
python -m app.eval.audit                 # separate real errors from strictness
python -m uvicorn app.main:app --port 8000
```

---

## Appendix C — Numbers at a glance

| | |
|---|---|
| Source manuals | 7 (+1 catalog, excluded) |
| Pages | ~230 |
| Searchable passages | 274 |
| Passages carrying warnings | 22 |
| Evaluation questions | 104 (70 answerable + 34 abstention) |
| Automated tests | 118 |
| Application code | ~5,100 lines |
| Test code | ~1,100 lines |
| Models running locally | 3 of 4 |
| Disk for models | ~2.5 GB |
| GPU required | none (phases 0–6) |
| RAM required | ~8 GB |
| Running cost | zero |
| End-to-end latency | ~2.9 s |

---

## Appendix D — Caveats on the numbers

Read before presenting:

- **Phase 3 figures are from the run completed 2026-10-11.** Entity-aware
  retrieval and heading recovery landed afterwards and are **not** reflected
  — a re-run is pending. Do not claim improvements that have not been
  measured.
- **0.862 is accuracy counting contradictions only.** The strict judge gives
  0.603. Always say which you are quoting and why.
- **Retrieval ceiling is 0.914, not 1.0**, because 6 of 70 quotes cannot be
  matched against the parser's output.
- **Abstention precision 0.588 understates current behaviour.** Several
  "leaks" now return per-entity labelled answers rather than one silently
  chosen value, which is arguably better than refusing — but the eval set
  scores them as failures. That is a product judgment worth settling.
- **MiniLM is English-only** and the corpus has German sections. The
  English-only eval set cannot detect that weakness.
