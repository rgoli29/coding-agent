"""Agent ABC and AgentResult.

MiniAgent (A) and GraphAgent (B) take the SAME constructor args and return the
SAME AgentResult. The runner never imports either one directly — it builds
whatever `agent.type` names. That is the A/B seam.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.config import AgentConfig
from src.dataset import Instance
from src.env.base import Environment
from src.history import History
from src.llm.client import GatewayClient
from src.tools.base import Tool


@dataclass
class AgentResult:
    instance_id: str
    patch: str
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    wall_clock_s: float = 0.0
    exit_status: str = "unknown"  # submitted | max_steps | error
    error: str = ""

    @property
    def has_patch(self) -> bool:
        return bool(self.patch.strip())


class Agent(ABC):
    def __init__(
        self,
        gateway: GatewayClient,
        tools: list[Tool],
        env: Environment,
        history: History,
        config: AgentConfig,
    ) -> None:
        self.gateway = gateway
        self.tools = tools
        self.env = env
        self.history = history
        self.config = config

    @abstractmethod
    def run(self, task: Instance) -> AgentResult: ...
