"""FastAPI application entrypoint.

Wires up the API gateway: auth, rate limits, sessions, request logging.
Stateless — holds no query or ingestion state itself, only routes to the
query pipeline (app.query) and ingestion pipeline (app.ingestion).

Models load once at startup, not per request. BGE-M3 takes the better
part of a minute to come up on CPU and the corpus index has to be built
or read from cache; doing that per request would put a 50-second floor
under a 3-second budget. The cost is a slow boot, which is the right
place to pay it.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import health, ingest, query

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_env()

    from app.eval.run import load_or_build
    from app.models.embedder.local import LocalEmbedder
    from app.models.generator.api import ApiGenerator
    from app.models.reranker.local import LocalReranker
    from app.query.pipeline import QueryPipeline
    from app.query.retrieve import InMemoryIndex

    weights = str(ROOT / "models" / "weights")

    corpus = load_or_build()
    index = InMemoryIndex()
    index.add(corpus.chunks, corpus.vectors)

    embedder = LocalEmbedder(cache_dir=weights)
    embedder.embed_query_sync("warm up")

    reranker = LocalReranker(
        os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        cache_dir=weights,
    )
    reranker.rerank_sync("warm up", ["warm up"])

    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    base_url = (
        os.environ.get("GROQ_BASE_URL")
        if os.environ.get("GROQ_API_KEY")
        else os.environ.get("OPENROUTER_BASE_URL", "")
    )
    generator = (
        ApiGenerator(os.environ.get("GENERATOR_MODEL", ""), api_key, base_url or "")
        if api_key
        else None
    )

    app.state.pipeline = QueryPipeline(
        index=index,
        embedder=embedder,
        reranker=reranker,
        generator=generator,
        score_floor=float(os.environ.get("RETRIEVAL_SCORE_FLOOR", "-5.0")),
        show_sources_floor=float(os.environ.get("SHOW_SOURCES_FLOOR", "-8.0")),
        top_k=int(os.environ.get("TOP_K", "5")),
    )
    app.state.stats = {
        "chunks": len(corpus.chunks),
        "documents": len(corpus.stats["documents"]),
        "embedder": corpus.model,
        "reranker": reranker.name,
        "generator": getattr(generator, "name", None),
    }
    yield


app = FastAPI(title="Installation & Commissioning Assistant", lifespan=lifespan)
app.include_router(query.router)
app.include_router(health.router)
app.include_router(ingest.router)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")
