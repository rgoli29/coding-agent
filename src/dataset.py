"""SWE-bench instances and subset selection.

`Instance` is the one data type shared by every layer (runner, env, agent,
evaluator), so it lives here and everything imports it from here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import DatasetConfig

log = logging.getLogger("dataset")

# Dataset shorthand -> HuggingFace dataset name.
DATASETS = {
    "swe-bench-lite": "princeton-nlp/SWE-bench_Lite",
    "swe-bench-verified": "princeton-nlp/SWE-bench_Verified",
    "swe-bench": "princeton-nlp/SWE-bench",
}


@dataclass
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    environment_setup_commit: str = ""
    version: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def hf_dataset_name(name: str) -> str:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose one of: {', '.join(DATASETS)}"
        ) from None


def load_instances(config: DatasetConfig) -> list[Instance]:
    """Load the configured subset.

    `instance_ids` wins when set (pin a few SAME-repo instances to keep the
    docker image footprint small); otherwise take the first `limit` rows.
    """
    from datasets import load_dataset

    name = hf_dataset_name(config.name)
    log.info("loading %s [%s]", name, config.split)
    rows = load_dataset(name, split=config.split)

    if config.instance_ids:
        wanted = set(config.instance_ids)
        selected = [r for r in rows if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in selected}
        if missing:
            raise ValueError(f"instance_ids not in {name}/{config.split}: {sorted(missing)}")
        # Preserve the order the user listed them in.
        order = {iid: i for i, iid in enumerate(config.instance_ids)}
        selected.sort(key=lambda r: order[r["instance_id"]])
    else:
        selected = list(rows.select(range(min(config.limit, len(rows)))))

    return [to_instance(r) for r in selected]


def to_instance(row: dict[str, Any]) -> Instance:
    return Instance(
        instance_id=row["instance_id"],
        repo=row["repo"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        environment_setup_commit=row.get("environment_setup_commit", ""),
        version=str(row.get("version", "")),
        raw=dict(row),
    )
