"""Step 5 — document version management.

A new revision of an existing manual retires the old chunks before the
new ones are indexed. This is what stops the assistant quoting a
superseded manual (failure mode 3). Retired chunks stay in storage for
audit but are filtered out of retrieval by default.
"""
