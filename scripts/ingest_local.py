"""Ingest the local corpus in Documents/ without going through the API.

Development convenience for phase 1, when there is no upload endpoint and
no UI yet. Runs the same worker code path as a queued job so that what is
tested here is what runs in production.
"""
