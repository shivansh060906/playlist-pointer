"""Answer generation over retrieved excerpts. Groq and Gemini both expose OpenAI-compatible endpoints,
so one tiny httpx client covers both (no vendor SDKs)."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence

import httpx

from .config import Settings
from .retrieval.hybrid import Hit
from .utils import format_timestamp

PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

SYSTEM_PROMPT = """You are a helpful assistant for a video playlist: {topic}. Answer the viewer's question using ONLY the numbered transcript excerpts provided.

Rules:
- Cite every claim with the excerpt number in square brackets, e.g. [1] or [2][3].
- If the question is unrelated to this playlist's subject or the excerpts are clearly not about it, reply with exactly NOT_COVERED and nothing else.
- If the excerpts only partly contain the answer, say so plainly. Do not fill gaps from outside knowledge.
- Transcripts are auto-generated and may contain misheard technical terms (e.g. "see out" for cout). Use the correct term silently.
- Be concise: at most ~150 words of plain prose. Include a short code snippet only if an excerpt clearly describes one.
- Mention which video and timestamp the viewer should jump to for the full explanation."""

_CITATION = re.compile(r"\[(\d+)\]")


class LLMNotConfigured(RuntimeError):
    pass


@dataclass
class LLMClient:
    provider: str
    api_key: str
    model: str
    timeout: float = 60.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        provider = settings.llm_provider
        if provider not in PROVIDERS:
            raise LLMNotConfigured(f"LLM_PROVIDER must be one of {list(PROVIDERS)}, got {provider!r}")
        key = settings.groq_api_key if provider == "groq" else settings.gemini_api_key
        if not key:
            raise LLMNotConfigured(
                f"No API key for provider '{provider}'. Set {provider.upper()}_API_KEY in backend/.env"
            )
        model = settings.groq_model if provider == "groq" else settings.gemini_model
        return cls(provider=provider, api_key=key, model=model)

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1200) -> str:
        url = f"{PROVIDERS[self.provider]}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error = "unknown error"
        for attempt in range(4):  # free tiers rate-limit aggressively: back off and retry
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"LLM request failed after retries ({last_error})")


def build_messages(question: str, hits: Sequence[Hit], topic: str = "a video series") -> list[dict]:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        blocks.append(f"[{i}] \"{c.title}\" (starts at {format_timestamp(c.start)})\n{c.text}")
    user = f"Question: {question}\n\nExcerpts:\n\n" + "\n\n".join(blocks)
    return [{"role": "system", "content": SYSTEM_PROMPT.format(topic=topic)}, {"role": "user", "content": user}]


def cited_indices(answer: str, n_hits: int) -> list[int]:
    """1-based excerpt numbers the model actually cited, in order of first use. Out-of-range numbers
    (hallucinated citations) are dropped rather than trusted."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= n_hits and number not in seen:
            seen.append(number)
    return seen
