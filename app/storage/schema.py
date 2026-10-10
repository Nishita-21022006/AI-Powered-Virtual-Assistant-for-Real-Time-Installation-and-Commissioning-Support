"""Table definitions.

Roughly:
  documents        source file, content hash, product, revision, status
  document_files   object-storage pointers: original, extracted figures
  chunks           extracted text, section, page, type, has_warning,
                   boundaries — stored independently of any vector
  embeddings       vector per (chunk, embedding model), pgvector column
  keyword_index    lexical index backing the exact-match half of hybrid
                   retrieval
  jobs             ingestion job state, per file, idempotency keys
  sessions         conversation history for follow-up rewriting
  query_log        full trace per query: see app.observability

Chunks and embeddings are deliberately separate tables, keyed by model,
so a re-embed writes new rows without touching parsed text.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# An HNSW index needs a fixed column width, so the dimension is pinned here
# rather than taken from the active embedder. 1536 is what the cloud
# candidates in CLAUDE.md section 6 can all emit (Gemini Embedding 001 and
# Voyage both support truncation to it). Changing it is a migration plus a
# full re-embed — which is exactly the operation the chunks/embeddings split
# exists to make possible.
EMBEDDING_DIM = 1536

DOCUMENT_STATUSES = ("pending", "active", "retired", "failed")
FILE_KINDS = ("original", "extracted_text", "figure")
CONTENT_TYPES = ("prose", "procedure", "table", "parameters", "figure_caption", "faq")
JOB_STATUSES = ("queued", "running", "succeeded", "failed")


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict] = {dict[str, Any]: JSONB}


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    product: Mapped[str] = mapped_column(String(128), nullable=False)
    revision: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    page_count: Mapped[int | None] = mapped_column(Integer)
    meta: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    files: Mapped[list[DocumentFile]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(status.in_(DOCUMENT_STATUSES), name="ck_documents_status"),
        UniqueConstraint("product", "revision", "content_hash", name="uq_documents_revision"),
        Index("ix_documents_content_hash", "content_hash"),
        Index("ix_documents_product_status", "product", "status"),
    )


class DocumentFile(Base):
    __tablename__ = "document_files"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="files")

    __table_args__ = (
        CheckConstraint(kind.in_(FILE_KINDS), name="ck_document_files_kind"),
        UniqueConstraint("object_key", name="uq_document_files_object_key"),
        Index("ix_document_files_document_id", "document_id"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str | None] = mapped_column(Text)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False, default="prose")
    has_warning: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="chunks")
    embeddings: Mapped[list[Embedding]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )
    keyword_entry: Mapped[KeywordIndex | None] = relationship(
        back_populates="chunk", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        CheckConstraint(content_type.in_(CONTENT_TYPES), name="ck_chunks_content_type"),
        CheckConstraint("char_end > char_start", name="ck_chunks_boundaries"),
        UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        # Retrieval reads live chunks only; retired ones are kept for audit.
        Index(
            "ix_chunks_live",
            "document_id",
            postgresql_where=retired_at.is_(None),
        ),
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False, default=EMBEDDING_DIM)
    vector: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("chunk_id", "model", name="uq_embeddings_chunk_model"),
        Index("ix_embeddings_model", "model"),
        Index(
            "ix_embeddings_vector_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )


class KeywordIndex(Base):
    __tablename__ = "keyword_index"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    tsv: Mapped[str] = mapped_column(TSVECTOR, nullable=False)

    chunk: Mapped[Chunk] = relationship(back_populates="keyword_entry")

    __table_args__ = (
        Index("ix_keyword_index_tsv", "tsv", postgresql_using="gin"),
        Index("ix_keyword_index_document_id", "document_id"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    stage: Mapped[str | None] = mapped_column(String(32))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(status.in_(JOB_STATUSES), name="ck_jobs_status"),
        UniqueConstraint("idempotency_key", name="uq_jobs_idempotency_key"),
        Index("ix_jobs_status_created_at", "status", "created_at"),
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_ref: Mapped[str | None] = mapped_column(String(128))
    product_scope: Mapped[str | None] = mapped_column(String(128))
    turns: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_sessions_user_ref", "user_ref"),)


class QueryLog(Base):
    __tablename__ = "query_log"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL")
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text)
    filters: Mapped[dict[str, Any]] = mapped_column(default=dict)
    retrieved: Mapped[dict[str, Any]] = mapped_column(default=dict)
    top_score: Mapped[float | None] = mapped_column(Float)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    answer: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[dict[str, Any]] = mapped_column(default=dict)
    embedder_model: Mapped[str | None] = mapped_column(String(128))
    reranker_model: Mapped[str | None] = mapped_column(String(128))
    generator_model: Mapped[str | None] = mapped_column(String(128))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_query_log_created_at", "created_at"),
        Index("ix_query_log_session_id", "session_id"),
    )


metadata = Base.metadata
