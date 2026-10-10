"""GET /jobs/{job_id} — ingestion job status.

Status is per file, not per batch, per the architecture's upload edge
cases: if two of three uploaded files fail, the caller sees exactly which.
"""
