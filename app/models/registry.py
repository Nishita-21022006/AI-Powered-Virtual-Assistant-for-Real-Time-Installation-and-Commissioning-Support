"""Backend selection — maps config to concrete implementations.

Reads the *_BACKEND settings and returns the matching implementation for
each role. The only place in the codebase that knows which backend is
active. Also owns the generator fallback chain (cloud -> local ->
retrieval-only) described in app.query.escalate.
"""
