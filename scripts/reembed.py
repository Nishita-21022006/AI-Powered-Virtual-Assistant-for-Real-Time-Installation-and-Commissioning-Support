"""Re-embed the corpus with a different embedding model.

Reads stored chunk text and boundaries, writes new vectors tagged with
the new model, and flips the index over once the batch completes. Never
re-parses source PDFs — that separation is the whole reason this script
is cheap to run, and the reason the storage layer is shaped the way it is.
"""
