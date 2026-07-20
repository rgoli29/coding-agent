"""Tool ABC — shared by Approach A's BashTool and Approach B's structured tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.env.base import Environment


@dataclass
class Observation:
    content: str
    exit_code: int = 0
    truncated: bool = False


class Tool(ABC):
    name: str
    spec: dict  # description + argument schema, rendered into the prompt

    @abstractmethod
    def run(self, args: dict, env: Environment) -> Observation: ...
