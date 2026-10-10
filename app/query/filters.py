"""Stage 4 — metadata filtering before retrieval.

Narrows the candidate set to the product line in session scope and to
current document revisions only. Retired chunks are excluded by default
(failure mode 3: answering from a superseded manual). This is also where
per-integrator document access control would attach if open question 6
comes back as "no, not everyone sees everything".
"""
