"""Environment ABC — the sandbox seam.

Everything above this interface (tools, agents, runner) is unaware of Docker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.dataset import Instance


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def output(self) -> str:
        """Combined stream, which is what a shell user would actually see."""
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)


class Environment(ABC):
    @abstractmethod
    def reset(self, instance: Instance) -> None:
        """Provision a fresh sandbox for `instance` at its base commit."""

    @abstractmethod
    def exec(self, cmd: str, timeout: int) -> ExecResult:
        """Run a shell command inside the sandbox."""

    @abstractmethod
    def read_file(self, path: str) -> str: ...

    @abstractmethod
    def write_file(self, path: str, content: str) -> None: ...

    @abstractmethod
    def get_diff(self) -> str:
        """`git diff` against the base commit — this is the submitted patch."""

    @abstractmethod
    def close(self) -> None:
        """Tear down the sandbox. Must be safe to call twice."""
