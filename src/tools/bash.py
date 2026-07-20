"""BashTool — Approach A's single tool.

One tool, one argument: a shell command run inside the task sandbox. The model
explores with grep/cat, edits with the shell, and runs the tests itself.
"""

from __future__ import annotations

from src.env.base import Environment
from src.tools.base import Observation, Tool

# Observations are truncated before they enter history, keeping head and tail:
# the head shows what the command started doing, the tail shows how it ended
# (which is where the error message lives).
MAX_OBSERVATION_CHARS = 10_000


def truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    omitted = len(text) - limit
    return f"{head}\n\n... [{omitted} chars omitted] ...\n\n{tail}", True


class BashTool(Tool):
    name = "bash"
    spec = {
        "name": "bash",
        "description": (
            "Run a shell command inside the task sandbox, from /testbed. "
            "State persists between calls."
        ),
        "arguments": {"cmd": "the shell command to run"},
    }

    def __init__(self, timeout_s: int = 120) -> None:
        self.timeout_s = timeout_s

    def run(self, args: dict, env: Environment) -> Observation:
        cmd = args.get("cmd", "").strip()
        if not cmd:
            return Observation(content="error: empty command", exit_code=1)

        result = env.exec(cmd, timeout=self.timeout_s)
        content, truncated = truncate(result.output)
        if not content.strip():
            content = "(no output)"
        return Observation(
            content=f"exit code: {result.exit_code}\n{content}",
            exit_code=result.exit_code,
            truncated=truncated,
        )
