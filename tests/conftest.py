"""Shared fixtures.

Integration tests run against a real Postgres with pgvector, not a mock —
vector search behaviour is the thing under test and a mock would prove
nothing about it.
"""
