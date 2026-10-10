"""Ingestion worker process — step 4.

Drains the queue and runs parse -> chunk -> embed -> tag for one document
at a time. Jobs are idempotent and retryable: a worker dying mid-job
leaves no half-indexed document visible (failure mode 11). Re-running
ingestion is always safe.

Runs as its own process, deployed separately from the API. For the MVP
this may be a single worker rather than a pool — the boundary is what
matters, not the scale.
"""
