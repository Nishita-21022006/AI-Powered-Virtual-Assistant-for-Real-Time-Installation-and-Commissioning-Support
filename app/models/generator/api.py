"""Generator over an OpenAI-compatible chat API.

One implementation, any provider. Groq, OpenRouter and a self-hosted
vLLM or Ollama endpoint all speak the same wire format, so switching is
a base URL and a key in config — which is the property the architecture
depends on for the phase 7 move in-house.

Why stdlib urllib and not the `openai` SDK
------------------------------------------
openai 3.x is built on `httpx2`, and on this machine httpx2 2.12.0
raises RecursionError inside SSL context setup on any request at all,
including a bare GET to an unrelated host. That breaks the SDK for
every provider, so it is an environment fault rather than something the
code can route around. urllib is stdlib, has no such problem, and the
request shape here is small enough that the SDK was buying little.
If the environment is ever fixed, swapping this for the SDK is a change
behind this class and nothing above it moves.

Groq sits behind Cloudflare, which rejects the default urllib
User-Agent with 403 "error code: 1010" — hence the browser UA. That is
a bot-protection fingerprint check, not authentication: the same key
succeeds the moment the header is present.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

from app.models.generator.base import Generation, Passage

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

SYSTEM_PROMPT = """You answer installation and commissioning questions for \
equipment installers, using ONLY the passages supplied.

An installer is reading this on site with the job stopped. Give them \
what they need to act, in the shape the question calls for.

HARD RULES, in order of importance:
1. Use only the passages. Never use prior knowledge about this or any \
other product.
2. Cite the source in square brackets after EVERY claim, exactly as the \
passage is labelled. This includes a one-line answer: a bare value with \
no citation is discarded and the installer sees nothing. Every line \
that states a fact carries its own citation.
3. Never state a number, measurement, tolerance or part code that does \
not appear verbatim in a passage.
4. If a passage carries a safety warning relevant to your answer, you \
MUST reproduce it. Never give a procedure without its warnings.
5. If the passages do not contain the answer, reply with exactly: \
NOT IN THESE DOCUMENTS

SHAPE THE ANSWER TO THE QUESTION:
- "How do I..." / a procedure -> a numbered list of steps, one action \
per line, in order. Do not merge steps into a paragraph.
- A specification, limit or value -> lead with the value and its \
citation on one line, then any condition that qualifies it.
- Wiring or terminals -> name each terminal and what connects to it, \
one per line.
- "Can I..." / compatibility -> start with Yes or No, then the \
condition the documents state.
- Anything else -> two or three short sentences.

FORMAT:
- Use "- " for bullets and "1. " for numbered steps. No other markup.
- Put any safety warning on its own final line, starting with \
"WARNING: ". Reproduce its wording from the passage.
- No preamble. Do not restate the question. Start with the answer."""

ABSTAIN_MARKER = "NOT IN THESE DOCUMENTS"


class GeneratorError(RuntimeError):
    """The provider could not be reached or refused the request.

    Raised rather than returned so the caller can fall through the
    chain — provider, then self-hosted, then retrieval-only — instead of
    mistaking an outage for a model that had nothing to say.
    """


@dataclass
class RateLimited(GeneratorError):
    retry_after: float | None = None


MAX_PASSAGE_CHARS = 1000
"""Per-passage cap sent to the generator.

Chunks run up to ~9,600 characters in this corpus, and five of those
exceeds the free tier's tokens-per-minute allowance outright — the API
returns 413 and the query degrades to sources-only.

1,000 is sized against the measured limit (8,000 tokens/minute) rather
than chosen for elegance: five passages at this cap plus the system
prompt is roughly 1,650 tokens, which sustains about four queries a
minute with headroom. It costs little, because the median chunk in
this corpus is 664 characters and is not truncated at all — only the
longest tenth is affected.

Warnings are appended AFTER the truncation point rather than being cut
with the rest of the text: losing the body of a passage costs detail,
losing its warning costs the guarantee.
"""


def format_passages(passages: Sequence[Passage]) -> str:
    lines = []
    for p in passages:
        label = p.doc_id
        if p.section:
            label += f", {p.section}"
        if p.page is not None:
            label += f", p{p.page}"
        body = p.text
        if len(body) > MAX_PASSAGE_CHARS:
            body = body[:MAX_PASSAGE_CHARS].rsplit(" ", 1)[0] + " …"
        lines.append(f"[{label}] {body}")
        for warning in p.warnings:
            lines.append(f"[{label}] WARNING: {warning}")
    return "\n\n".join(lines)


class ApiGenerator:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        max_tokens: int = 400,
    ):
        self.name = model
        self.model = model
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._max_tokens = max_tokens

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self._base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            if exc.code == 429:
                retry = exc.headers.get("retry-after")
                raise RateLimited(float(retry) if retry else None) from exc
            if exc.code == 402:
                raise GeneratorError(
                    "402 from the provider. On OpenRouter this means a negative "
                    f"balance, which blocks even free models. Detail: {detail}"
                ) from exc
            raise GeneratorError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise GeneratorError(f"{type(exc).__name__}: {exc}") from exc

    MAX_RETRY_SLEEP = 30.0
    """Ceiling on a server-supplied retry-after.

    `retry-after` is a number the provider chooses, and honouring it
    without a bound means an outside party can stall the process for as
    long as it likes. A 104-question evaluation run hung for minutes on
    a single long value with no log line and no CPU activity, which
    looks like a crash rather than a wait. Past this ceiling, fail and
    let the caller fall back to retrieval-only.
    """

    def _post_with_retry(self, payload: dict, attempts: int = 3) -> dict:
        """Retry on rate limiting, honouring retry-after up to a cap.

        The free tier is ~30 requests/minute, and an evaluation run of a
        hundred questions will hit it. Giving up on the first 429 makes
        the whole query degrade to sources-only, which reads as a
        quality problem rather than a quota one.
        """
        delay = 2.0
        for attempt in range(attempts):
            try:
                return self._post(payload)
            except RateLimited as exc:
                if attempt == attempts - 1:
                    raise
                wait = min(exc.retry_after or delay, self.MAX_RETRY_SLEEP)
                time.sleep(wait)
                delay *= 2
        raise GeneratorError("unreachable")

    def generate_sync(
        self, question: str, passages: Sequence[Passage]
    ) -> Generation:
        started = time.perf_counter()
        data = self._post_with_retry(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Passages:\n{format_passages(passages)}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                "max_tokens": self._max_tokens,
                "temperature": 0,
            }
        )
        text = (data["choices"][0]["message"].get("content") or "").strip()
        abstained = text.upper().startswith(ABSTAIN_MARKER)

        return Generation(
            text=text,
            cited_doc_ids=() if abstained else tuple({p.doc_id for p in passages}),
            model=data.get("model", self.model),
            abstained=abstained,
        )

    async def generate(
        self, question: str, passages: Sequence[Passage]
    ) -> Generation:
        return self.generate_sync(question, passages)
