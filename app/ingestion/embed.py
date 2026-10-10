"""Embedding — turn each chunk into a searchable vector.

Calls the embedder through app.models.embedder. Writes vectors to a
store keyed by chunk id, keeping them separate from the extracted text
and chunk boundaries so that changing embedding model is a batch
re-embed rather than a re-parse of every PDF.
"""
