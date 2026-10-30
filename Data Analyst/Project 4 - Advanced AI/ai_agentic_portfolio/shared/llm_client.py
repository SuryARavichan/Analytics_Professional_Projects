"""
llm_client.py — a small, pluggable LLM client shared by all three AI
portfolio projects (QueryMind, EchoIQ, SentinelBI).

WHY THIS EXISTS
----------------
Each project needs to call a generative LLM (to write SQL, summarize a
customer-review cluster, or draft an incident report). Rather than pasting
API calls into every notebook, this module gives every project the same
interface — swap the backend once, every project picks it up.

TWO BACKENDS
------------
1. LiveClient    — calls the real OpenAI or Anthropic API using your own
                    key (reads OPENAI_API_KEY or ANTHROPIC_API_KEY from the
                    environment). This is the "production" path.
2. CachedClient  — replays pre-recorded, hand-authored responses from a
                    local JSON fixture file. This is the default, so every
                    notebook in this portfolio runs end-to-end, offline,
                    free, and reproducibly — no API key required to review
                    the work. Every cached response in this portfolio was
                    authored by grounding it in the *actual* output of the
                    real, executed analysis upstream (the real SQL result,
                    the real cluster's top TF-IDF terms, the real anomaly
                    stats) — it is not a fabricated example disconnected
                    from the data.

get_client() auto-selects: LiveClient if an API key is present in the
environment, otherwise CachedClient. This means the exact same notebook
code runs in demo mode today and in live mode the moment you export a key
— that swap is the whole point of the abstraction.

USAGE
-----
    from llm_client import get_client
    client = get_client(cache_file="cached_responses.json")
    response = client.complete(prompt_id="insight_orders_by_region",
                                system="You are a data analyst...",
                                prompt="Given this data: ...")
"""
from __future__ import annotations
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMResponse:
    text: str
    backend: str
    model: str
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "", prompt_id: str | None = None) -> LLMResponse:
        ...


class CachedClient(LLMClient):
    """Offline, zero-cost, reproducible backend. Looks up a pre-authored
    response by `prompt_id` from a JSON fixture file. Falls back to a
    clearly-labeled generic stub if an id isn't found, so the pipeline
    never silently breaks -- it just tells you the fixture is missing."""

    def __init__(self, cache_file: str):
        self.cache_file = Path(cache_file)
        if self.cache_file.exists():
            self._cache = json.loads(self.cache_file.read_text())
        else:
            self._cache = {}

    def complete(self, prompt: str, system: str = "", prompt_id: str | None = None) -> LLMResponse:
        t0 = time.time()
        if prompt_id and prompt_id in self._cache:
            text = self._cache[prompt_id]
        else:
            text = ("[CACHED-MODE STUB] No pre-recorded response found for "
                    f"prompt_id={prompt_id!r}. Add a real API key to run this "
                    "live, or add a fixture to cached_responses.json.")
        return LLMResponse(
            text=text, backend="cached", model="cached-fixture",
            prompt_tokens_est=len(prompt.split()), completion_tokens_est=len(text.split()),
            latency_ms=(time.time() - t0) * 1000, cached=True,
        )


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def complete(self, prompt: str, system: str = "", prompt_id: str | None = None) -> LLMResponse:
        t0 = time.time()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.2)
        text = resp.choices[0].message.content
        usage = resp.usage
        return LLMResponse(
            text=text, backend="openai", model=self.model,
            prompt_tokens_est=usage.prompt_tokens, completion_tokens_est=usage.completion_tokens,
            latency_ms=(time.time() - t0) * 1000, cached=False,
        )


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, prompt: str, system: str = "", prompt_id: str | None = None) -> LLMResponse:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024, system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return LLMResponse(
            text=text, backend="anthropic", model=self.model,
            prompt_tokens_est=resp.usage.input_tokens, completion_tokens_est=resp.usage.output_tokens,
            latency_ms=(time.time() - t0) * 1000, cached=False,
        )


def get_client(cache_file: str = "cached_responses.json", force: str | None = None) -> LLMClient:
    """Auto-selects a backend. `force` can be 'live', 'cached', 'openai', or
    'anthropic' to override auto-detection."""
    if force == "cached":
        return CachedClient(cache_file)
    if force == "openai" or (force != "cached" and os.environ.get("OPENAI_API_KEY") and force != "anthropic"):
        return OpenAIClient()
    if force == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    return CachedClient(cache_file)
