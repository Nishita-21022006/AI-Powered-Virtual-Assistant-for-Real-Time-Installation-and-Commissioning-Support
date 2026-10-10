"""Structured logging setup.

Every query and ingestion job logs through this, feeding the observability
layer described in the architecture (section 3): retrieved chunks, scores,
model used, latency, verdict, feedback.
"""
