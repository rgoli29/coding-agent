"""MiniAgent — Approach A.

A while loop with one tool. Nothing is hidden: the model sees the raw shell and
drives the whole task itself, and the patch is whatever `git diff` says at the
end. Modeled on mini-swe-agent.
"""

from __future__ import annotations

import logging
import re
import time

from src.agents.base import Agent, AgentResult
from src.dataset import Instance
from src.llm.client import GatewayError
from src.tools.base import Observation

log = logging.getLogger("agent")

SUBMIT_COMMAND = "submit"

SYSTEM_PROMPT = """\
You are a software engineer fixing a real bug in a checked-out repository.

You work by issuing ONE shell command at a time and reading its output. The repo
is at /testbed and it is your working directory. State persists between commands.

Respond with exactly one bash code block and nothing that matters outside it:

```bash
your command here
```

Rules:
- ONE command per response. To chain, use && or ; inside the single block.
- Commands are non-interactive. Never run anything that waits for input, opens a
  pager, or runs forever (no vim, no git commit, no `python` REPL, no watch).
- Use `grep -rn`, `find`, `cat -n`, `sed -n '10,40p' file` to read code.
- Edit files with `python - <<'PY' ... PY` heredocs or `sed -i`. Prefer small,
  surgical edits. Do not rewrite whole files.
- Do not modify tests. Fix the source.
- When the fix is complete and you have verified it, respond with exactly:

```bash
submit
```

Your changes are collected as a `git diff`, so just edit files in place. Do not
create a patch file, do not commit, do not push.

Strategy: locate the relevant code, read enough context to be sure, make the
smallest correct change, then run the project's own tests to check it.
"""

TASK_PROMPT = """\
Repository: {repo}
Instance: {instance_id}

Fix the issue described below. Start by locating the relevant code.

--- ISSUE ---
{problem_statement}
--- END ISSUE ---

Begin. Respond with a single bash code block.
"""

FORMAT_REMINDER = (
    "Your response contained no bash code block. Respond with exactly one:\n\n"
    "```bash\nyour command here\n```\n\n"
    "or ```bash\nsubmit\n``` when the fix is complete."
)

BASH_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)


def parse_action(completion: str) -> str | None:
    """Extract the command from the LAST bash block. None if there is none.

    Last, not first: models often show a wrong-then-corrected command, and the
    final block is the one they intend to run.
    """
    blocks = BASH_BLOCK.findall(completion)
    if not blocks:
        return None
    command = blocks[-1].strip()
    return command or None


def is_submit(action: str) -> bool:
    return action.strip().lower() == SUBMIT_COMMAND


class MiniAgent(Agent):
    def run(self, task: Instance) -> AgentResult:
        started = time.monotonic()
        result = AgentResult(instance_id=task.instance_id, patch="")

        self.history.reset(
            system_prompt=SYSTEM_PROMPT,
            task_prompt=TASK_PROMPT.format(
                repo=task.repo,
                instance_id=task.instance_id,
                problem_statement=task.problem_statement,
            ),
        )

        tool = self.tools[0]  # Approach A has exactly one tool: bash

        for step in range(1, self.config.max_steps + 1):
            result.steps = step
            try:
                completion = self.gateway.complete(self.history.messages())
            except GatewayError as exc:
                log.error("[%s] gateway error: %s", task.instance_id, exc)
                result.exit_status = "error"
                result.error = str(exc)
                break

            result.prompt_tokens += completion.prompt_tokens
            result.completion_tokens += completion.completion_tokens

            action = parse_action(completion.content)
            if action is None:
                log.warning("[%s] step %d: no bash block", task.instance_id, step)
                self.history.append(completion.content, Observation(content=FORMAT_REMINDER))
                continue

            if is_submit(action):
                log.info("[%s] step %d: submit", task.instance_id, step)
                result.exit_status = "submitted"
                break

            log.info("[%s] step %d: %s", task.instance_id, step, action.replace("\n", " ")[:120])
            observation = tool.run({"cmd": action}, self.env)
            self.history.append(completion.content, observation)

            # max_cost_usd is enforced here; real cost figures arrive with the
            # gateway's accounting in M3, so today this only trips if configured.
            if self.config.max_cost_usd > 0 and result.cost_usd >= self.config.max_cost_usd:
                result.exit_status = "max_cost"
                break
        else:
            log.warning("[%s] out of steps (%d)", task.instance_id, self.config.max_steps)
            result.exit_status = "max_steps"

        result.patch = self.env.get_diff()
        result.wall_clock_s = round(time.monotonic() - started, 1)
        log.info(
            "[%s] done: %s, %d steps, %.1fs, patch %d chars",
            task.instance_id,
            result.exit_status,
            result.steps,
            result.wall_clock_s,
            len(result.patch),
        )
        return result
