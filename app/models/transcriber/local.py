"""Transcriber — NVIDIA Parakeet TDT 0.6B v2, local, CPU, free, ~0.7 GB.
Phase 6.

Open weights, so no NVIDIA API account is needed. Small, fast,
commercially licensed, and — unlike Whisper — it does not hallucinate
text during silence, which matters on a noisy site. faster-whisper is
the fallback implementation.

Variants if requirements change: v3 for 25 European languages at ~0.3
WER points, Whisper large-v3 if CJK is ever needed (Parakeet omits it),
Nemotron Speech Streaming if push-to-talk ever becomes live streaming.

The benchmark figures behind this choice come from NVIDIA's own
published numbers and two third-party comparisons with differing
conditions — re-check before committing, and read the OpenMDW licence
properly if the Nemotron route is taken.
"""
