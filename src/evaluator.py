"""Evaluator — wraps the official SWE-bench harness.

We shell out to `python -m swebench.harness.run_evaluation` rather than calling
into it: the harness manages its own docker lifecycle and process pool, and an
out-of-process failure cannot take the run down with it.

An instance counts as RESOLVED only if every FAIL_TO_PASS test now passes and
every PASS_TO_PASS test still passes — the harness decides that, not us.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from src.config import Config
from src.dataset import hf_dataset_name

log = logging.getLogger("evaluator")


def evaluate(config: Config, predictions_path: Path, run_id: str) -> dict:
    """Run the harness over `predictions_path`. Returns the parsed report dict."""
    if not predictions_path.exists() or not predictions_path.read_text().strip():
        log.warning("no predictions at %s — skipping evaluation", predictions_path)
        return {}

    run_dir = predictions_path.parent
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", hf_dataset_name(config.dataset.name),
        "--split", config.dataset.split,
        "--predictions_path", str(predictions_path.resolve()),
        "--max_workers", str(config.eval.max_workers),
        "--cache_level", config.eval.cache_level,
        "--clean", str(config.eval.clean),
        "--run_id", run_id,
        "--report_dir", str(run_dir.resolve()),
        "--timeout", str(config.sandbox.task_timeout_s),
    ]
    log.info("running the SWE-bench harness (this takes a while)")
    log.debug("harness cmd: %s", " ".join(cmd))

    # cwd=run_dir so the harness's own logs/ tree lands in the bind-mounted run
    # directory and survives the container.
    proc = subprocess.run(cmd, cwd=run_dir, text=True)
    if proc.returncode != 0:
        log.error("harness exited %d — see the logs under %s", proc.returncode, run_dir)

    report = find_report(run_dir, run_id)
    if not report:
        log.error("no harness report found in %s", run_dir)
    return report


def find_report(run_dir: Path, run_id: str) -> dict:
    """The harness writes <model_name>.<run_id>.json into report_dir."""
    candidates = sorted(run_dir.glob(f"*{run_id}.json"))
    if not candidates:
        return {}
    try:
        report = json.loads(candidates[-1].read_text())
    except json.JSONDecodeError:
        log.error("harness report %s is not valid JSON", candidates[-1])
        return {}
    report["_report_path"] = str(candidates[-1])
    return report
