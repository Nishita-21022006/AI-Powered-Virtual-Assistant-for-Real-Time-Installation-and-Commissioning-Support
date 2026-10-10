"""Step 6 — index the new chunks and invalidate the cache.

Cache invalidation is not optional: a stale cached answer served after
re-indexing (failure mode 15) defeats the entire version manager. The
index is itself versioned by embedding model, and queries against a
mixed-model index are blocked (failure mode 17).
"""
