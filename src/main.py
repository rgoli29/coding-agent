"""Entrypoint — runs INSIDE the agent container.

Preflight (gateway health + docker socket), print the run plan, then run the
pipeline. Never touches an API key: the model is reachable only through the
native gateway on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.config import Config, load_config
from src.llm.client import GatewayClient

log = logging.getLogger("main")

RUNS_DIR = Path("runs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="coding-agent")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--run-id", default=None, help="reuse an existing run id")
    parser.add_argument("--plan-only", action="store_true", help="preflight + plan, then exit")
    parser.add_argument("--eval-only", action="store_true", help="re-evaluate existing predictions")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report only")
    return parser.parse_args(argv)


def check_docker() -> tuple[bool, str]:
    """The runner spawns sibling containers via the mounted host socket."""
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker CLI unavailable: {exc}"
    if proc.returncode != 0:
        return False, f"docker info failed: {proc.stderr.strip()[:200]}"
    return True, f"docker daemon {proc.stdout.strip()}"


def preflight(config: Config) -> bool:
    ok = True

    client = GatewayClient(config)
    if client.health():
        log.info("[ok]   gateway healthy at %s", config.gateway.base_url)
    else:
        log.error(
            "[FAIL] no gateway at %s — start it on the host with `make gateway`",
            config.gateway.base_url,
        )
        ok = False
    client.close()

    docker_ok, detail = check_docker()
    log.info("%s %s", "[ok]  " if docker_ok else "[FAIL]", detail)
    ok = ok and docker_ok

    return ok


def print_plan(config: Config, run_dir: Path) -> None:
    ds = config.dataset
    subset = ", ".join(ds.instance_ids) if ds.instance_ids else f"first {ds.limit} of {ds.split}"
    lines = [
        "",
        "run plan",
        "--------",
        f"  run dir      {run_dir}",
        f"  dataset      {ds.name} [{subset}]",
        f"  agent        {config.agent.type}  (A=mini, B=graph)",
        f"  model        {config.model.name} @ temp {config.model.temperature}",
        f"  budgets      max_steps={config.agent.max_steps} "
        f"max_cost_usd={config.agent.max_cost_usd} step_timeout_s={config.agent.step_timeout_s}",
        f"  sandbox      task_timeout_s={config.sandbox.task_timeout_s}",
        f"  eval         run_eval={config.eval.run_eval} workers={config.eval.max_workers} "
        f"cache_level={config.eval.cache_level} clean={config.eval.clean}",
        "",
    ]
    print("\n".join(lines), flush=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    config = load_config(args.config)

    from src.evaluator import evaluate
    from src.runner import load_results, run_all
    from src.telemetry import write_report

    # --eval-only / --report-only reuse an existing run instead of making one.
    reuse = args.eval_only or args.report_only
    if reuse:
        run_dir = RUNS_DIR / args.run_id if args.run_id else latest_run_dir()
        if run_dir is None:
            log.error("no previous run found under %s/", RUNS_DIR)
            return 1
        run_id = run_dir.name
        log.info("reusing run %s", run_dir)
    else:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        results = load_results(run_dir)
        from src.evaluator import find_report

        path = write_report(run_dir, results, find_report(run_dir, run_id))
        log.info("report written to %s", path)
        return 0

    if args.eval_only:
        docker_ok, detail = check_docker()  # the harness needs the socket too
        if not docker_ok:
            log.error("[FAIL] %s", detail)
            return 1
        results = load_results(run_dir)
    else:
        if not preflight(config):
            return 1
        print_plan(config, run_dir)
        if args.plan_only:
            return 0
        results = run_all(config, run_dir)
        summarize(results, run_dir)

    report = {}
    if config.eval.run_eval or args.eval_only:
        report = evaluate(config, run_dir / "predictions.jsonl", run_id)

    path = write_report(run_dir, results, report)
    log.info("report written to %s", path)
    print_headline(results, report)
    return 0


def latest_run_dir() -> Path | None:
    runs = sorted((d for d in RUNS_DIR.glob("*") if d.is_dir()), key=lambda d: d.name)
    return runs[-1] if runs else None


def print_headline(results: list, report: dict) -> None:
    resolved = set(report.get("resolved_ids", []))
    hits = len([r for r in results if r.instance_id in resolved])
    total = len(results) or 1
    print("", flush=True)
    print(f"RESOLVED {hits}/{len(results)} ({hits / total * 100:.1f}%)", flush=True)


def summarize(results: list, run_dir: Path) -> None:
    with_patch = sum(1 for r in results if r.has_patch)
    steps = sum(r.steps for r in results)
    tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)
    print("", flush=True)
    print(f"{len(results)} instance(s), {with_patch} with a non-empty patch", flush=True)
    print(f"{steps} agent steps, {tokens} tokens", flush=True)
    print(f"predictions: {run_dir / 'predictions.jsonl'}", flush=True)
    for r in results:
        mark = "patch" if r.has_patch else "EMPTY"
        detail = f" ({r.error})" if r.error else ""
        print(
            f"  {r.instance_id:<40} {r.exit_status:<10} {mark:<6} "
            f"{r.steps:>3} steps {r.wall_clock_s:>7.1f}s{detail}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
