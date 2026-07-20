"""GatewayClient — the runner's thin HTTP client for the native gateway.

The runner reaches the model ONLY through this client, and this client only
ever talks to http://127.0.0.1:8000. No API key ever enters this process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import Config


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Completion:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class GatewayError(RuntimeError):
    """The gateway returned a non-200, or a body we could not read."""


class GatewayClient:
    def __init__(self, config: Config, timeout_s: float = 300.0) -> None:
        self.config = config
        self.base_url = config.gateway.base_url
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_s))

    def health(self) -> bool:
        try:
            return self.client.get(f"{self.base_url}/health").status_code == 200
        except httpx.HTTPError:
            return False

    def complete(self, messages: list[Message]) -> Completion:
        payload = {
            "model": self.config.model.name,
            "temperature": self.config.model.temperature,
            "messages": [m.to_dict() for m in messages],
        }
        try:
            resp = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise GatewayError(f"could not reach gateway at {self.base_url}: {exc}") from exc

        if resp.status_code != 200:
            raise GatewayError(f"gateway returned {resp.status_code}: {resp.text[:500]}")

        body = resp.json()
        try:
            content = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise GatewayError(f"malformed completion: {str(body)[:500]}") from exc

        usage = body.get("usage") or {}
        return Completion(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            raw=body,
        )

    def close(self) -> None:
        self.client.close()
