"""The upload queue — step 3 of the ingestion path.

Absorbs upload spikes so nothing blocks. The API enqueues and returns a
job id immediately; a separate worker pool drains the queue. The two
pools scale independently, which is what guarantees that thirty files
uploaded at once can never degrade a live query (failure mode 10).
"""
