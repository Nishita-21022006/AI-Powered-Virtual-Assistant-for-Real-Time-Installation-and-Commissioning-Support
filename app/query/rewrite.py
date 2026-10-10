"""Stage 3 — query rewriting against session history.

Turns "and the other one?" into a standalone question. Mitigates failure
mode 9 (follow-up question loses its referent). Needs session memory from
app.storage.sessions.
"""
