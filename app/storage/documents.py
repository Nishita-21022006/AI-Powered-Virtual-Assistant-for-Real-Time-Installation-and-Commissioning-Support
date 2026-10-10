"""Document and revision repository.

Owns the supersede path used by the ingestion version manager: retire the
chunks of the previous revision, index the new ones, keep the retired
rows for audit while excluding them from retrieval.
"""
