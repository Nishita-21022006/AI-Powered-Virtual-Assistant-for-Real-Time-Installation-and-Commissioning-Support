"""Guardrail tier 4 — escalation, and the degraded-mode fallback chain.

When the gate abstains or verification rejects: say confidence is low,
show the most relevant sections anyway, name the document that would
cover it, offer the support path.

The fallback chain is cloud generator -> local generator -> retrieval-only
passages. Even with every model unavailable, the system returns the right
sections with citations. It never returns nothing.
"""
