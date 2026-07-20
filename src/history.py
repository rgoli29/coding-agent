"""History — the conversation, trimmed to fit a token budget.

Shared by both approaches. The system prompt and the task statement are pinned;
when the budget is exceeded the OLDEST step pairs are dropped first, because the
recent ones carry the state the model is actually reasoning about.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.client import Message
from src.tools.base import Observation

# Rough chars-per-token. Deliberately conservative: over-estimating tokens makes
# us trim early, which is far cheaper than a context-overflow error.
CHARS_PER_TOKEN = 4


@dataclass
class Step:
    action: str
    observation: str

    def messages(self) -> list[Message]:
        return [
            Message(role="assistant", content=self.action),
            Message(role="user", content=self.observation),
        ]

    def char_len(self) -> int:
        return len(self.action) + len(self.observation)


class History:
    def __init__(
        self,
        system_prompt: str = "",
        task_prompt: str = "",
        token_budget: int = 60_000,
    ) -> None:
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        self.token_budget = token_budget
        self.steps: list[Step] = []
        self.dropped = 0

    def reset(self, system_prompt: str, task_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        self.steps = []
        self.dropped = 0

    def append(self, action: str, observation: Observation | str) -> None:
        content = observation.content if isinstance(observation, Observation) else observation
        self.steps.append(Step(action=action, observation=content))

    def messages(self) -> list[Message]:
        """The prompt, trimmed to the budget. Pinned head + most recent steps."""
        pinned = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self.task_prompt),
        ]
        budget_chars = self.token_budget * CHARS_PER_TOKEN
        used = sum(len(m.content) for m in pinned)

        kept: list[Step] = []
        for step in reversed(self.steps):
            if used + step.char_len() > budget_chars and kept:
                break
            used += step.char_len()
            kept.append(step)
        kept.reverse()

        self.dropped = len(self.steps) - len(kept)
        out = list(pinned)
        if self.dropped:
            out.append(
                Message(
                    role="user",
                    content=f"[{self.dropped} earlier step(s) dropped to fit the context budget]",
                )
            )
        for step in kept:
            out.extend(step.messages())
        return out

    def to_dict(self) -> dict:
        """Serialised trajectory, written per instance for debugging."""
        return {
            "system_prompt": self.system_prompt,
            "task_prompt": self.task_prompt,
            "steps": [{"action": s.action, "observation": s.observation} for s in self.steps],
            "dropped": self.dropped,
        }
