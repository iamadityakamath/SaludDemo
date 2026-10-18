"""Minimal client for an OpenAI-compatible chat endpoint (Groq, team-hosted vLLM/Ollama/llama.cpp).

Configuration comes from the environment or a repo-root .env file (gitignored), so no URL or
key lands in git. Real environment variables win over .env.

    GROQ_API_KEY         Groq key; selects https://api.groq.com/openai/v1 unless a base URL is set
    GROQ_MODEL           Groq model id (default openai/gpt-oss-120b)
    SALUD_LLM_BASE_URL   any other OpenAI-compatible server  (default http://localhost:11434/v1)
    SALUD_LLM_MODEL      model id for that server (overrides GROQ_MODEL)
    SALUD_LLM_API_KEY    bearer token for that server (overrides GROQ_API_KEY)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://localhost:11434/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines into os.environ without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str) -> str | None:
    load_dotenv()
    return os.environ.get(name) or None


def _default_base_url() -> str:
    return _env("SALUD_LLM_BASE_URL") or (GROQ_BASE_URL if _env("GROQ_API_KEY") else DEFAULT_BASE_URL)


def _default_model() -> str | None:
    if _env("SALUD_LLM_MODEL"):
        return _env("SALUD_LLM_MODEL")
    if _env("GROQ_API_KEY") and not _env("SALUD_LLM_BASE_URL"):
        return _env("GROQ_MODEL") or GROQ_DEFAULT_MODEL
    return None


def _default_key() -> str | None:
    return _env("SALUD_LLM_API_KEY") or _env("GROQ_API_KEY")


@dataclass
class LLMEndpoint:
    base_url: str = field(default_factory=_default_base_url)
    model: str | None = field(default_factory=_default_model)
    api_key: str | None = field(default_factory=_default_key, repr=False)
    timeout: int = 300
    max_retries: int = 3

    @property
    def is_local(self) -> bool:
        return urlparse(self.base_url).hostname in LOCAL_HOSTS

    @property
    def provider(self) -> str:
        host = urlparse(self.base_url).hostname or ""
        return "Groq" if host.endswith("groq.com") else ("local" if self.is_local else host)

    def _request(self, path: str, body: dict | None = None) -> dict:
        # A real User-Agent: some hosted APIs reject urllib's default one.
        headers = {"Content-Type": "application/json", "User-Agent": "salud-pipeline/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.base_url.rstrip("/") + path, headers=headers, data=data)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries:
                    wait = float(e.headers.get("retry-after") or 2 ** attempt)
                    time.sleep(min(wait, 30))
                    continue
                detail = e.read().decode(errors="replace")[:300]
                raise RuntimeError(f"LLM endpoint returned HTTP {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                raise RuntimeError(f"Cannot reach LLM endpoint {self.base_url}: {e.reason}") from None
        raise RuntimeError("unreachable")

    def chat(self, messages: list[dict], temperature: float = 0.0, json_mode: bool = False,
             max_tokens: int | None = None) -> str:
        """max_tokens matters on Groq: without it, the rate limiter reserves a large default per request."""
        if not self.model:
            raise ValueError("No model set: pass --model or set SALUD_LLM_MODEL / GROQ_MODEL")
        body = {"model": self.model, "temperature": temperature, "messages": messages}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._request("/chat/completions", body)["choices"][0]["message"]["content"]

    def list_models(self) -> list[str]:
        return [m["id"] for m in self._request("/models").get("data", [])]
