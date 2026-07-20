# CLAUDE.md — conventions & invariants

A SWE-bench coding agent. The spec is `coding-agent-design.md`; read it before
changing architecture. This file is the short version for future sessions.

## Topology (do not change)

| Piece | Where |
|---|---|
| Gateway (`src/llm/gateway_server.py`) | **native host process**, `127.0.0.1:8000`, holds the API key |
| Runner + agent + tools + evaluator + telemetry | **container** (`docker compose` service `agent`), `network_mode: host` |
| Per-task sandboxes + SWE-bench tests | sibling containers on the host daemon via the mounted `/var/run/docker.sock` |
| Model inference | free hosted API, reached **only** through the native gateway |

## Hard invariants

1. The runner depends only on the ABCs: `Agent`, `Tool`, `Environment`,
   `GatewayClient`, `History`. Never on concrete classes. `runner.build_agent`
   resolves `agent.type` through `AGENT_REGISTRY` by lazy import.
2. Approach A → B is a `config.yaml` change (`agent.type: mini` → `graph`) plus
   **new files only** (`src/agents/graph.py`, `src/tools/structured.py`).
   No edits to runner / env / gateway / evaluator / telemetry.
3. The API key lives **only** in the native gateway process. The agent container
   never receives it (`.env` is not mounted into the container).
4. The runner reaches the model **only** via the gateway on `127.0.0.1:8000`.

## Environment

- Python 3.11+, managed with [uv](https://docs.astral.sh/uv/). Host: `make setup`
  (`uv venv` + `uv pip install -e .`). The container installs with uv, no venv.
- Dependencies: `pydantic` v2, `pyyaml`, `httpx`, `fastapi`, `uvicorn`,
  `python-dotenv`, `datasets`, `swebench`. Approach B adds `langgraph` (extra
  `graph`). Ask before adding anything heavier.

## Commands

```bash
make setup          # host venv + .env
make gateway        # start native gateway, wait for /health
make plan           # preflight + print run plan (no model calls)
make run            # gateway + containerized pipeline
make eval           # re-run evaluator over existing predictions
make report         # rebuild report from the last run
make stop-gateway
make clean          # docker prune + drop runs/
```

## Sandbox execution rules (SWE-bench images)

- Work in `/testbed`.
- Non-login `bash -c`, so set `BASH_ENV=/root/.bashrc` — **without it `conda
  activate testbed` never runs and every test fails.**
- Clean output env: `PAGER=cat`, `MANPAGER=cat`, `LESS=-R`, `PIP_PROGRESS_BAR=off`,
  `TQDM_DISABLE=1`.
- Truncate observations (~10k chars, head+tail) before they enter history.
- The patch is `git -C /testbed diff` at submit time.

## Resource limits (16 GB RAM host)

Keep `eval.max_workers: 1`, `cache_level: env`, `clean: true`, and pin instances
to as few distinct repos as possible — environment images are the disk cost.

## Build order

M0 skeleton → M1 agent resolves one instance → M2 subset+eval+report →
M3 gateway hardening → M4 improve A → M5 Approach B. Don't start a milestone
before the previous one meets the acceptance criteria in the spec §16.
