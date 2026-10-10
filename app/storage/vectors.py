"""Vector store access (pgvector).

Vectors are tagged with the embedding model that produced them. The
index version is checked on every query; a mixed-model index blocks
queries rather than silently returning incoherent neighbours.
"""
