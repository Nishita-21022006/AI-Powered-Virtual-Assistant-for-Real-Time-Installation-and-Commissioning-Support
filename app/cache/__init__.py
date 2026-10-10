"""Semantic answer cache — query stage 2.

The same questions recur constantly across integrators on a job. A cache
hit costs ~20ms against ~1.5s for a full generation, which changes the
concurrency arithmetic more than any serving tweak.

Invalidated on every index change, without exception: a stale answer
served after a manual revision defeats the version manager entirely
(failure mode 15).
"""
