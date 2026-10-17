"""Minimal client for an OpenAI-compatible chat endpoint (team-hosted vLLM/Ollama/llama.cpp).

Configuration comes from the environment so no URL or key lands in git:
    SALUD_LLM_BASE_URL   e.g. https://llm.team.example/v1   (default http://localhost:11434/v1)
    SALUD_LLM_MODEL      e.g. qwen2.5:7b-instruct
    SALUD_LLM_API_KEY    bearer token, if the server requires one
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEFAULT_BASE_URL = "http://localhost:11434/v1"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class LLMEndpoint:
    base_url: str = field(default_factory=lambda: os.environ.get("SALUD_LLM_BASE_URL", DEFAULT_BASE_URL))
    model: str | None = field(default_factory=lambda: os.environ.get("SALUD_LLM_MODEL"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("SALUD_LLM_API_KEY"), repr=False)
    timeout: int = 300

    @property
    def is_local(self) -> bool:
        return urlparse(self.base_url).hostname in LOCAL_HOSTS

    def _request(self, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url.rstrip("/") + path, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"LLM endpoint returned HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach LLM endpoint {self.base_url}: {e.reason}") from None

    def chat(self, messages: list[dict], temperature: float = 0.0, json_mode: bool = False) -> str:
        if not self.model:
            raise ValueError("No model set: pass --model or set SALUD_LLM_MODEL")
        body = {"model": self.model, "temperature": temperature, "messages": messages}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._request("/chat/completions", body)["choices"][0]["message"]["content"]

    def list_models(self) -> list[str]:
        return [m["id"] for m in self._request("/models").get("data", [])]
