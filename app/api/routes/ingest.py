"""POST /documents — document upload.

The architecture's target design is asynchronous: accept the file,
return a job id immediately, and let a separate worker pool parse,
chunk, embed and index it, so an upload spike can never degrade a live
query. That separation is phase 5.

What runs here now is the honest MVP version of it — the upload is
processed inline and the caller waits. It is correct but not yet
isolated, so a large upload WILL slow concurrent queries. The route
shape, the job id and the per-file status are already in place, so
phase 5 is moving the work behind a queue rather than redesigning the
interface.

Per-file status, not per-batch: if two of three files fail the caller
sees exactly which, per the architecture's upload edge cases.
"""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from pydantic import BaseModel

from app.ingestion.chunk import chunk_document
from app.ingestion.parse import ParseError, parse_pdf
from app.query.retrieve import IndexedChunk

router = APIRouter(tags=["documents"])

MAX_BYTES = 50 * 1024 * 1024


class FileResult(BaseModel):
    filename: str
    status: str
    detail: str = ""
    chunks: int = 0
    warnings: int = 0


class UploadResponse(BaseModel):
    job_id: str
    files: list[FileResult]
    total_chunks_now: int


@router.get("/documents")
async def list_documents(request: Request) -> dict:
    index = request.app.state.pipeline.index
    docs: dict[str, dict] = {}
    for chunk in index.chunks:
        d = docs.setdefault(chunk.document, {"chunks": 0, "warnings": 0})
        d["chunks"] += 1
        d["warnings"] += int(chunk.has_warning)
    return {"documents": docs, "total_chunks": len(index.chunks)}


@router.post("/documents", response_model=UploadResponse)
async def upload_documents(
    request: Request, files: list[UploadFile] = File(...)
) -> UploadResponse:
    pipeline = request.app.state.pipeline
    index = pipeline.index
    embedder = pipeline.embedder

    existing_hashes = getattr(request.app.state, "content_hashes", None)
    if existing_hashes is None:
        existing_hashes = {}
        request.app.state.content_hashes = existing_hashes

    results: list[FileResult] = []
    new_chunks: list[IndexedChunk] = []
    new_texts: list[str] = []

    for upload in files:
        name = upload.filename or "unnamed"
        if not name.lower().endswith(".pdf"):
            results.append(FileResult(filename=name, status="rejected", detail="not a PDF"))
            continue

        payload = await upload.read()
        if len(payload) > MAX_BYTES:
            results.append(
                FileResult(
                    filename=name,
                    status="rejected",
                    detail=f"{len(payload) // 1024 // 1024} MB exceeds the "
                    f"{MAX_BYTES // 1024 // 1024} MB cap",
                )
            )
            continue

        digest = hashlib.sha256(payload).hexdigest()
        if digest in existing_hashes:
            results.append(
                FileResult(
                    filename=name,
                    status="skipped",
                    detail=f"identical content already indexed as {existing_hashes[digest]}",
                )
            )
            continue

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        try:
            parsed = parse_pdf(tmp_path, extract_figure_images=False)
        except ParseError as exc:
            results.append(FileResult(filename=name, status="failed", detail=str(exc)))
            continue
        finally:
            tmp_path.unlink(missing_ok=True)

        chunks = chunk_document(parsed)
        if not chunks:
            results.append(
                FileResult(
                    filename=name,
                    status="failed",
                    detail="no indexable text found — the file may be a scan needing OCR",
                )
            )
            continue

        # A re-upload of a manual already present supersedes it: the old
        # chunks are dropped so the assistant cannot quote a retired
        # revision alongside the new one.
        superseded = sum(1 for c in index.chunks if c.document == name)
        for c in chunks:
            new_chunks.append(
                IndexedChunk(
                    chunk_id=f"{name}#{c.ordinal}",
                    document=name,
                    text=c.text,
                    page=c.page_start,
                    section=c.section,
                    has_warning=c.has_warning,
                )
            )
            new_texts.append(c.text)
        existing_hashes[digest] = name
        results.append(
            FileResult(
                filename=name,
                status="indexed",
                chunks=len(chunks),
                warnings=sum(c.has_warning for c in chunks),
                detail=f"superseded {superseded} chunks of a previous revision"
                if superseded
                else "",
            )
        )

    if new_chunks:
        replaced = {c.document for c in new_chunks}
        kept = [c for c in index.chunks if c.document not in replaced]
        kept_vectors = [
            index.vectors[i].tolist()
            for i, c in enumerate(index.chunks)
            if c.document not in replaced
        ]
        new_vectors = embedder.embed_documents_sync(new_texts, batch_size=8)
        index.add(kept + new_chunks, kept_vectors + new_vectors)

    return UploadResponse(
        job_id=str(uuid.uuid4()),
        files=results,
        total_chunks_now=len(index.chunks),
    )
