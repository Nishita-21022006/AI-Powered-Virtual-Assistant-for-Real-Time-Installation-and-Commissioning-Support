"""Generator from self-hosted weights — phase 7, the offline requirement.

Qwen 3 at 4B or 8B, the same family as the OpenRouter model, so prompts
carry over unchanged. Served through an OpenAI-compatible endpoint
(Ollama in development, vLLM in production) so selecting it is a config
change.

This is the only role that needs a GPU, and therefore the only reason
phase 7 needs hardware at all — the embedder, reranker and transcriber
are already local and run on CPU.
"""
