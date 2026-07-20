"""The native LLM gateway.

Runs on the HOST (never in a container) and is the only process that ever sees
the API key. Speaks the OpenAI chat-completions shape so the runner's client
stays trivial.

    GET  /health              -> 200 {"status": "ok", ...}
    POST /v1/chat/completions -> proxied upstream

M0: pure passthrough. Rate limiting, caching, retry and cost logging land in M3.

Run it with:  python -m src.llm.gateway_server
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import Config, load_config
from src.llm.providers import Provider, get_provider

log = logging.getLogger("gateway")


class Gateway:
    """Holds the upstream client. State that M3 adds (buckets, cache) lives here."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.provider: Provider = get_provider(config.gateway.upstream)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180.0))

    async def complete(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            resp = await self.client.post(
                self.provider.chat_completions_url,
                headers=self.provider.headers(),
                json=payload,
            )
        except httpx.HTTPError as exc:
            return 502, {"error": {"message": f"transport error: {exc}"}}
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"error": {"message": resp.text[:500]}}

    async def aclose(self) -> None:
        await self.client.aclose()


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="coding-agent gateway")
    gateway = Gateway(config)
    app.state.gateway = gateway

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "upstream": gateway.provider.name,
            "model": config.model.name,
        }

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
    # Fail fast and loudly if the key is missing, rather than on the first call.
    get_provider(config.gateway.upstream).api_key()
    log.info(
        "gateway listening on %s:%s -> %s (%s)",
        config.gateway.host,
        config.gateway.port,
        config.gateway.upstream,
        config.model.name,
    )
    uvicorn.run(
        create_app(config),
        host=config.gateway.host,
        port=config.gateway.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
