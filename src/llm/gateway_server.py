"""The native LLM gateway.

Runs on the HOST (never in a container) and is the only process that ever sees
the API key. Speaks the OpenAI chat-completions shape so the runner's client
stays trivial.

    GET  /health              -> 200 {"status": "ok", ...}
    GET  /stats               -> counters for this gateway process
    POST /v1/chat/completions -> proxied upstream, with cache + rate limit + retry

M3 hardening, all inside the Gateway class so nothing upstream of it changes:
  - token-bucket rate limiting on requests/min AND tokens/min (tpm usually binds)
  - a response cache keyed on (model, messages, temperature), persisted to disk
  - exponential backoff on 429 / 5xx / transport errors
  - a per-call log of tokens + estimated cost

The cache and call log live in a STABLE directory (runs/gateway/ by default),
not a per-run one: the gateway is one long-lived process serving many runs, and
the cache only pays off when it survives across them — a re-run of the same eval
then replays from disk and costs nothing upstream.

Run it with:  python -m src.llm.gateway_server
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import Config, load_config
from src.llm.providers import Provider, get_provider

log = logging.getLogger("gateway")

# Rough chars->tokens ratio, used only to RESERVE tpm budget before a call. The
# response's real usage reconciles the estimate afterwards.
CHARS_PER_TOKEN = 4


class TokenBucket:
    """Classic token bucket: `capacity` units, refilled evenly over `per_seconds`."""

    def __init__(self, capacity: float, per_seconds: float = 60.0) -> None:
        self.capacity = max(1.0, float(capacity))
        self.per_seconds = per_seconds
        self.tokens = self.capacity
        self.updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        rate = self.capacity / self.per_seconds
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * rate)
        self.updated = now

    async def acquire(self, amount: float) -> float:
        """Block until `amount` units are available. Returns seconds waited."""
        amount = min(amount, self.capacity)  # never deadlock on an oversized ask
        waited = 0.0
        while True:
            self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return waited
            deficit = amount - self.tokens
            delay = deficit / (self.capacity / self.per_seconds)
            waited += delay
            await asyncio.sleep(min(delay, 5.0))

    def give_back(self, amount: float) -> None:
        self._refill()
        self.tokens = min(self.capacity, self.tokens + max(0.0, amount))


class ResponseCache:
    """Cache keyed on (model, messages, temperature), persisted as one JSON file."""

    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.entries: dict[str, Any] = {}
        if enabled and path.exists():
            try:
                self.entries = json.loads(path.read_text())
                log.info("loaded %d cached completion(s) from %s", len(self.entries), path)
            except json.JSONDecodeError:
                log.warning("cache file %s is corrupt; starting empty", path)

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        material = json.dumps(
            {
                "model": payload.get("model"),
                "messages": payload.get("messages"),
                "temperature": payload.get("temperature"),
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key) if self.enabled else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries))
        tmp.replace(self.path)  # atomic: a crash mid-write can't corrupt the cache


class Gateway:
    def __init__(self, config: Config, state_dir: Path) -> None:
        self.config = config
        self.provider: Provider = get_provider(config.gateway.upstream)
        self.state_dir = state_dir
        state_dir.mkdir(parents=True, exist_ok=True)

        self.requests = TokenBucket(config.gateway.requests_per_minute)
        self.tokens = TokenBucket(config.gateway.tokens_per_minute)
        self.cache = ResponseCache(state_dir / "cache.json", config.gateway.cache)
        self.log_path = state_dir / "calls.jsonl"
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "upstream_calls": 0,
            "retries": 0,
            "rate_limited": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
        }

    # -- logging / cost ----------------------------------------------------

    def _log_call(self, record: dict[str, Any]) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        m = self.config.model
        return (
            prompt_tokens * m.input_usd_per_1m + completion_tokens * m.output_usd_per_1m
        ) / 1_000_000

    def _estimate_prompt_tokens(self, payload: dict[str, Any]) -> int:
        chars = sum(len(str(m.get("content", ""))) for m in payload.get("messages", []))
        return max(1, chars // CHARS_PER_TOKEN)

    # -- the one entry point ----------------------------------------------

    async def complete(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.stats["requests"] += 1
        key = ResponseCache.key(payload)

        cached = self.cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            self._log_call({"event": "cache_hit", "key": key[:12], "model": payload.get("model")})
            return 200, cached

        # Reserve rate-limit budget BEFORE calling. tpm usually binds first.
        estimated = self._estimate_prompt_tokens(payload)
        await self.requests.acquire(1)
        await self.tokens.acquire(estimated)

        status, body = await self._call_upstream(payload)
        if status != 200:
            self.stats["errors"] += 1
            self._log_call({"event": "error", "key": key[:12], "status": status})
            return status, body

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        # Reconcile the pre-call reservation with what was actually spent.
        actual = prompt_tokens + completion_tokens
        if actual and actual < estimated:
            self.tokens.give_back(estimated - actual)
        elif actual > estimated:
            await self.tokens.acquire(actual - estimated)

        cost = self._cost(prompt_tokens, completion_tokens)
        self.stats["prompt_tokens"] += prompt_tokens
        self.stats["completion_tokens"] += completion_tokens
        self.stats["cost_usd"] = round(self.stats["cost_usd"] + cost, 6)
        self._log_call(
            {
                "event": "upstream",
                "key": key[:12],
                "model": payload.get("model"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(cost, 6),
            }
        )
        self.cache.put(key, body)
        return 200, body

    async def _call_upstream(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST upstream with exponential backoff on 429 / 5xx / transport errors."""
        delay = 2.0
        last: tuple[int, dict[str, Any]] = (502, {"error": {"message": "no attempt made"}})
        for attempt in range(self.config.gateway.max_retries + 1):
            try:
                self.stats["upstream_calls"] += 1
                resp = await self.client.post(
                    self.provider.chat_completions_url,
                    headers=self.provider.headers(),
                    json=payload,
                )
                if resp.status_code == 200:
                    return 200, resp.json()
                last = (resp.status_code, self._safe_json(resp))
                if resp.status_code == 429:
                    self.stats["rate_limited"] += 1
                elif resp.status_code < 500 and resp.status_code not in (408, 409):
                    return last  # a 4xx retrying cannot fix (bad key, bad model)
            except httpx.HTTPError as exc:
                last = (502, {"error": {"message": f"transport error: {exc}"}})

            if attempt < self.config.gateway.max_retries:
                self.stats["retries"] += 1
                sleep_s = self._retry_after(last) or delay
                self._log_call(
                    {"event": "retry", "attempt": attempt + 1, "status": last[0], "sleep_s": round(sleep_s, 2)}
                )
                await asyncio.sleep(sleep_s)
                delay = min(delay * 2, 60.0)
        return last

    @staticmethod
    def _retry_after(last: tuple[int, dict[str, Any]]) -> float | None:
        """Honour the provider's own 'try again in Ns' hint when it gives one."""
        if last[0] != 429:
            return None
        import re

        msg = str(last[1])
        m = re.search(r"try again in ([\d.]+)s", msg)
        return min(float(m.group(1)) + 0.5, 60.0) if m else None

    @staticmethod
    def _safe_json(resp: httpx.Response) -> dict[str, Any]:
        try:
            return resp.json()
        except ValueError:
            return {"error": {"message": resp.text[:500]}}

    async def aclose(self) -> None:
        await self.client.aclose()


def create_app(config: Config, state_dir: Path) -> FastAPI:
    app = FastAPI(title="coding-agent gateway")
    gateway = Gateway(config, state_dir)
    app.state.gateway = gateway

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "upstream": gateway.provider.name,
            "model": config.model.name,
            "cache": config.gateway.cache,
        }

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return gateway.stats

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        payload = await request.json()
        payload.setdefault("model", config.model.name)
        payload.setdefault("temperature", config.model.temperature)
        status, body = await gateway.complete(payload)
        return JSONResponse(status_code=status, content=body)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await gateway.aclose()

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    load_dotenv()
    config = load_config(os.environ.get("CONFIG_PATH", "config.yaml"))
    state_dir = Path(os.environ.get("GATEWAY_STATE_DIR", "runs/gateway"))
    # Fail fast and loudly if the key is missing, rather than on the first call.
    get_provider(config.gateway.upstream).api_key()
    log.info(
        "gateway listening on %s:%s -> %s (%s) | cache=%s rpm=%s tpm=%s | state=%s",
        config.gateway.host,
        config.gateway.port,
        config.gateway.upstream,
        config.model.name,
        config.gateway.cache,
        config.gateway.requests_per_minute,
        config.gateway.tokens_per_minute,
        state_dir,
    )
    uvicorn.run(
        create_app(config, state_dir),
        host=config.gateway.host,
        port=config.gateway.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
