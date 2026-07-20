"""Upstream adapters.

All three supported providers expose an OpenAI-compatible /chat/completions
endpoint, so an adapter is just (base_url, api-key env var). The key is read
here — inside the NATIVE gateway process — and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set. Copy .env.example to .env and fill it in."
            )
        return key

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key()}",
            "Content-Type": "application/json",
        }


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    ),
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
}


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown upstream {name!r}. Choose one of: {', '.join(PROVIDERS)}"
        ) from None
