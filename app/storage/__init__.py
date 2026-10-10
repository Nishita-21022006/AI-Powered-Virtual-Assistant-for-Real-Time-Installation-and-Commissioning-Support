"""Storage layer.

Postgres holds vectors (pgvector), the keyword index, chunk metadata and
document versions. Object storage holds original files and extracted
figure images.

The load-bearing separation: extracted text and chunk boundaries live
apart from the vectors derived from them. That is what makes an
embedding model change a batch re-embed instead of a re-parse of every
PDF in the corpus.
"""
