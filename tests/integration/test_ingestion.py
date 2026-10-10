"""Ingestion edge cases from architecture section 5.

Duplicate upload skipped by content hash; a new revision retires the old
chunks; a scanned image-only PDF routes to OCR and is flagged low
confidence; corrupt and password-protected files fail cleanly with a
readable reason; oversized files are rejected at upload rather than ten
minutes in; a worker killed mid-job leaves nothing half-indexed and the
retry is safe; two of three files failing reports per-file status.
"""
