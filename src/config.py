"""Config schema (pydantic v2) and loader for config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    # For the gateway's cost estimate. Free tier -> leave at 0 (cost logs as $0).
    input_usd_per_1m: float = 0.0
    output_usd_per_1m: float = 0.0


class GatewayConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    upstream: Literal["gemini", "groq", "openrouter"] = "gemini"
    requests_per_minute: int = 10
    tokens_per_minute: int = 250_000
    cache: bool = True
    max_retries: int = 5

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class AgentConfig(BaseModel):
    type: str = "mini"  # THE A/B switch: mini (A) | graph (B)
    max_steps: int = 40
    max_cost_usd: float = 0.0
    step_timeout_s: int = 120
    history_token_budget: int = 60_000


class DatasetConfig(BaseModel):
    name: str = "swe-bench-lite"
    split: str = "test"
    limit: int = 10
    instance_ids: list[str] = Field(default_factory=list)


class SandboxConfig(BaseModel):
    task_timeout_s: int = 900
    pull_timeout_s: int = 300


class EvalConfig(BaseModel):
    run_eval: bool = True
    max_workers: int = 1
    cache_level: Literal["none", "base", "env", "instance"] = "env"
    clean: bool = True


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # pydantic v2 protects the `model_` namespace; `model` is a config section here.
    model_config = {"protected_namespaces": ()}


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate config.yaml. Missing file -> all defaults."""
    path = Path(path)
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(raw)
