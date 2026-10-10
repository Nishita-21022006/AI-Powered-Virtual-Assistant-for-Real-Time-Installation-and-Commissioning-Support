"""Observability — full query tracing.

Every query is logged with its retrieved chunks, their scores, the model
used, latency per stage, the verdict (answered / abstained / rejected by
verification) and any user feedback.

This is one of the five properties of "production-ready MVP": for any
answer the system gave, we can reconstruct exactly which passages
produced it and why. It is also how failure mode 18 — retrieval quality
degrading silently over time — becomes visible.
"""
