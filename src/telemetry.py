"""Telemetry — metrics, failure taxonomy, and the run report.

The taxonomy is the point of this module: % resolved tells you how you did,
the taxonomy tells you what to fix next. Fix the biggest bucket, re-measure.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.agents.base import AgentResult

# Ordered: the first matching rule wins, most-specific first.
FAILURE_LABELS = {
    "resolved": "resolved",
    "api_error": "API / rate-limit error",
    "context_overflow": "context overflow",
    "sandbox_error": "sandbox error",
    "out_of_steps": "out of steps",
    "localization_miss": "localization miss (empty patch)",
    "malformed_patch": "malformed patch (harness could not apply/run it)",
    "tests_failing": "tests still failing",
    "unknown": "unknown",
}


def classify(result: AgentResult, report: dict) -> str:
    """Put one instance in exactly one bucket."""
    resolved = set(report.get("resolved_ids", []))
    if result.instance_id in resolved:
        return "resolved"

    if result.exit_status == "error":
        blob = result.error.lower()
        if "sandbox" in blob:
            return "sandbox_error"
        # Check rate limits BEFORE context: a 429 body says "tokens per minute",
        # which otherwise looks like a context-length complaint.
        if "429" in blob or "rate limit" in blob or "rate_limit" in blob:
            return "api_error"
        if "context length" in blob or "context window" in blob or "too long" in blob:
            return "context_overflow"
        return "api_error"

    if not result.has_patch:
        # No edit was ever made: the model never found the code to change.
        return "out_of_steps" if result.exit_status == "max_steps" else "localization_miss"

    if result.exit_status == "max_steps":
        return "out_of_steps"

    # A patch exists but did not resolve. The harness distinguishes "could not
    # apply / errored" from "applied but tests failed".
    if result.instance_id in set(report.get("error_ids", [])):
        return "malformed_patch"
    if result.instance_id in set(report.get("unresolved_ids", [])):
        return "tests_failing"
    if result.instance_id in set(report.get("empty_patch_ids", [])):
        return "localization_miss"
    return "unknown"


def summarize(results: list[AgentResult], report: dict) -> dict:
    resolved = set(report.get("resolved_ids", []))
    buckets: dict[str, list[str]] = {}
    per_instance = []

    for r in results:
        bucket = classify(r, report)
        buckets.setdefault(bucket, []).append(r.instance_id)
        per_instance.append(
            {
                **asdict(r),
                "patch_chars": len(r.patch),
                "resolved": r.instance_id in resolved,
                "bucket": bucket,
                # The patch itself lives in predictions.jsonl; keep the report readable.
                "patch": None,
            }
        )

    total = len(results) or 1
    return {
        "instances": len(results),
        "resolved": len([r for r in results if r.instance_id in resolved]),
        "resolve_rate": round(len([r for r in results if r.instance_id in resolved]) / total, 3),
        "with_patch": len([r for r in results if r.has_patch]),
        "total_steps": sum(r.steps for r in results),
        "total_prompt_tokens": sum(r.prompt_tokens for r in results),
        "total_completion_tokens": sum(r.completion_tokens for r in results),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
        "total_wall_clock_s": round(sum(r.wall_clock_s for r in results), 1),
        "buckets": buckets,
        "per_instance": per_instance,
        "harness_report": {k: v for k, v in report.items() if not isinstance(v, list)},
    }


def write_report(run_dir: Path, results: list[AgentResult], report: dict) -> Path:
    """Write report.json + report.md into the run directory. Returns the .md path."""
    summary = summarize(results, report)
    (run_dir / "report.json").write_text(json.dumps(summary, indent=2))

    md = [
        f"# Run report — `{run_dir.name}`",
        "",
        "## Headline",
        "",
        f"- **Resolved: {summary['resolved']}/{summary['instances']} "
        f"({summary['resolve_rate'] * 100:.1f}%)**",
        f"- Non-empty patches: {summary['with_patch']}/{summary['instances']}",
        f"- Steps: {summary['total_steps']} | "
        f"Tokens: {summary['total_prompt_tokens'] + summary['total_completion_tokens']} "
        f"(in {summary['total_prompt_tokens']} / out {summary['total_completion_tokens']}) | "
        f"Cost: ${summary['total_cost_usd']:.4f}",
        f"- Wall clock: {summary['total_wall_clock_s']:.0f}s",
        "",
        "## Per instance",
        "",
        "| instance | resolved | exit status | steps | patch chars | wall clock | bucket |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["per_instance"]:
        md.append(
            f"| `{row['instance_id']}` | {'✅' if row['resolved'] else '❌'} | "
            f"{row['exit_status']} | {row['steps']} | {row['patch_chars']} | "
            f"{row['wall_clock_s']:.0f}s | {FAILURE_LABELS.get(row['bucket'], row['bucket'])} |"
        )

    md += ["", "## Failure taxonomy", "", "| bucket | count | instances |", "|---|---|---|"]
    for bucket, ids in sorted(summary["buckets"].items(), key=lambda kv: -len(kv[1])):
        md.append(
            f"| {FAILURE_LABELS.get(bucket, bucket)} | {len(ids)} | "
            f"{', '.join(f'`{i}`' for i in ids)} |"
        )

    md += [
        "",
        "Fix the biggest non-resolved bucket, then re-run the same subset and compare.",
        "",
    ]

    if summary["harness_report"]:
        md += ["## Harness totals", "", "```json",
               json.dumps(summary["harness_report"], indent=2), "```", ""]

    path = run_dir / "report.md"
    path.write_text("\n".join(md))
    return path
