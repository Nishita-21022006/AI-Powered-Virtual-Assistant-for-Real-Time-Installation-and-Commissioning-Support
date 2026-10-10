"""Database engine and session management (SQLAlchemy + psycopg).

Separate connection pools for the API process and the ingestion workers,
sized independently — they have very different concurrency profiles and
must not contend.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Literal

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

Profile = Literal["api", "ingestion"]

# A query is synchronous and latency-budgeted, so the API pool is wide and
# refuses to wait long for a connection. Ingestion is async batch work on a
# small worker pool: few connections, held for the length of a parse, and
# content to queue.
_POOLS: dict[Profile, dict[str, int]] = {
    "api": {"pool_size": 10, "max_overflow": 10, "pool_timeout": 5},
    "ingestion": {"pool_size": 4, "max_overflow": 2, "pool_timeout": 60},
}


def database_url() -> str:
    return os.environ["DATABASE_URL"]


@lru_cache(maxsize=len(_POOLS))
def get_engine(profile: Profile = "api") -> Engine:
    return create_engine(
        database_url(),
        pool_pre_ping=True,
        pool_recycle=1800,
        **_POOLS[profile],
    )


@lru_cache(maxsize=len(_POOLS))
def get_sessionmaker(profile: Profile = "api") -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(profile), expire_on_commit=False)


@contextmanager
def session_scope(profile: Profile = "api") -> Iterator[Session]:
    with get_sessionmaker(profile)() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def api_session() -> Iterator[Session]:
    with session_scope("api") as session:
        yield session
