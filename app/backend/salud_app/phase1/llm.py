"""Minimal client for an OpenAI-compatible chat endpoint (Groq by default).

Settings come from the environment or the repository root .env (read by config.Settings):
    GROQ_API_KEY         Groq key; selects https://api.groq.com/openai/v1 unless a base URL is set
    GROQ_MODEL           Groq model id (default openai/gpt-oss-120b)
    SALUD_LLM_BASE_URL   any other OpenAI-compatible server (e.g. a local Ollama)
    SALUD_LLM_MODEL      model id for that server (overrides GROQ_MODEL)
    SALUD_LLM_API_KEY    bearer token for that server (overrides GROQ_API_KEY)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..config import get_settings

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _base_url() -> str:
    s = get_settings()
    return s.salud_llm_base_url or (GROQ_BASE_URL if s.groq_api_key else DEFAULT_BASE_URL)


def _model() -> str | None:
    s = get_settings()
    if s.salud_llm_model:
        return s.salud_llm_model
    if s.groq_api_key and not s.salud_llm_base_url:
        return s.groq_model or GROQ_DEFAULT_MODEL
    return None


def _key() -> str | None:
    s = get_settings()
    return s.salud_llm_api_key or s.groq_api_key


@dataclass
class LLMEndpoint:
    base_url: str = field(default_factory=_base_url)
    model: str | None = field(default_factory=_model)
    api_key: str | None = field(default_factory=_key, repr=False)
    timeout: int = 300
    max_retries: int = 3

    @property
    def is_local(self) -> bool:
        return urlparse(self.base_url).hostname in LOCAL_HOSTS

    @property
    def provider(self) -> str:
        host = urlparse(self.base_url).hostname or ""
        return "Groq" if host.endswith("groq.com") else ("local" if self.is_local else host)

    @property
    def ready(self) -> bool:
        return bool(self.model and (self.api_key or self.is_local))

    def _request(self, path: str, body: dict) -> dict:
        # A real User-Agent: some hosted APIs reject urllib's default one.
        headers = {"Content-Type": "application/json", "User-Agent": "salud-app/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.base_url.rstrip("/") + path, headers=headers, data=data)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries:
                    time.sleep(min(float(e.headers.get("retry-after") or 2 ** attempt), 30))
                    continue
                raise RuntimeError(f"LLM endpoint returned HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise RuntimeError(f"Cannot reach LLM endpoint {self.base_url}: {getattr(e, 'reason', e)}") from None
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict], temperature: float = 0.0, json_mode: bool = False,
             max_tokens: int | None = None) -> str:
        """max_tokens matters on Groq: without it, the rate limiter reserves a large default per request."""
        if not self.model:
            raise ValueError("No model set: set GROQ_API_KEY (and optionally GROQ_MODEL) in .env")
        body: dict = {"model": self.model, "temperature": temperature, "messages": messages}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._request("/chat/completions", body)["choices"][0]["message"]["content"]
