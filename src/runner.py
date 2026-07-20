"""Runner — orchestrates agent runs over the dataset subset.

INVARIANT: this module depends only on the ABCs in `agents.base`, `env.base`,
`tools.base`, `llm.client` and `history`. It must never import MiniAgent or
GraphAgent directly — `build_agent` resolves `agent.type` lazily so that adding
Approach B is a new file plus one config line.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.agents.base import Agent, AgentResult
from src.config import Config
from src.dataset import Instance, load_instances
from src.env.base import Environment
from src.history import History
from src.llm.client import GatewayClient
from src.tools.base import Tool

log = logging.getLogger("runner")

# agent.type -> "module:class". Approach B registers "graph" here and nothing else changes.
AGENT_REGISTRY: dict[str, str] = {
    "mini": "src.agents.mini:MiniAgent",
    "graph": "src.agents.graph:GraphAgent",
}


def build_agent(
    config: Config,
    gateway: GatewayClient,
    tools: list[Tool],
    env: Environment,
    history: History,
) -> Agent:
    """Resolve `agent.type` to a concrete Agent without importing it at module load."""
    from importlib import import_module

    try:
        target = AGENT_REGISTRY[config.agent.type]
    except KeyError:
        raise ValueError(
            f"Unknown agent.type {config.agent.type!r}. "
            f"Known types: {', '.join(AGENT_REGISTRY)}"
        ) from None

    module_name, class_name = target.split(":")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"agent.type={config.agent.type!r} needs {module_name}, which is not present: {exc}"
        ) from exc

    agent_cls = getattr(module, class_name)
    return agent_cls(gateway, tools, env, history, config.agent)


def build_tools(config: Config) -> list[Tool]:
    """Approach A gets exactly one tool. Approach B registers its own set here."""
    from src.tools.bash import BashTool

    return [BashTool(timeout_s=config.agent.step_timeout_s)]


def build_env(config: Config) -> Environment:
    from src.env.docker_env import DockerEnvironment

    return DockerEnvironment(config.sandbox)


def run_all(config: Config, run_dir: Path) -> list[AgentResult]:
    """Run the configured agent over the configured subset, one instance at a time."""
    instances = load_instances(config.dataset)
    log.info("running %d instance(s) with agent.type=%s", len(instances), config.agent.type)

    gateway = GatewayClient(config)
    env = build_env(config)
    results: list[AgentResult] = []

    predictions_path = run_dir / "predictions.jsonl"
    trajectory_dir = run_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("")  # fresh file per run

    try:
        for index, instance in enumerate(instances, start=1):
            log.info("=== [%d/%d] %s ===", index, len(instances), instance.instance_id)
            history = History(token_budget=config.agent.history_token_budget)
            result = run_one(config, instance, gateway, env, history)
            results.append(result)
            append_prediction(predictions_path, config, result)
            (trajectory_dir / f"{instance.instance_id}.json").write_text(
                json.dumps(history.to_dict(), indent=2)
            )
    finally:
        env.close()
        gateway.close()

    save_results(run_dir, results)
    return results


def save_results(run_dir: Path, results: list[AgentResult]) -> None:
    """Persist agent telemetry so --eval-only / --report-only can reuse it."""
    (run_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )


def load_results(run_dir: Path) -> list[AgentResult]:
    path = run_dir / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"no results.json in {run_dir} — run the pipeline first")
    return [AgentResult(**row) for row in json.loads(path.read_text())]


def run_one(
    config: Config,
    instance: Instance,
    gateway: GatewayClient,
    env: Environment,
    history: History,
) -> AgentResult:
    """One instance: provision the sandbox, run the agent, always tear down."""
    try:
        env.reset(instance)
    except Exception as exc:  # sandbox failures must not kill the whole run
        log.error("[%s] sandbox failed: %s", instance.instance_id, exc)
        return AgentResult(
            instance_id=instance.instance_id,
            patch="",
            exit_status="error",
            error=f"sandbox: {exc}",
        )

    agent = build_agent(config, gateway, build_tools(config), env, history)
    try:
        return agent.run(instance)
    except Exception as exc:
        log.exception("[%s] agent crashed", instance.instance_id)
        return AgentResult(
            instance_id=instance.instance_id,
            patch="",
            exit_status="error",
            error=f"agent: {exc}",
        )
    finally:
        env.close()


def append_prediction(path: Path, config: Config, result: AgentResult) -> None:
    """One SWE-bench prediction row per instance."""
    row = {
        "instance_id": result.instance_id,
        "model_name_or_path": config.model.name,
        "model_patch": result.patch,
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
