# Coding Agent

A SWE-bench coding agent that runs on your laptop, for free.

Give it a real GitHub issue and the repository it belongs to; it explores the
code in a sandbox, edits files, runs the tests, and submits a patch. The patch is
then scored by the official SWE-bench harness: an instance counts as **resolved**
only if the tests that were failing now pass *and* the tests that were passing
still do.

The point is not to beat the leaderboard. It is to build the scaffold with
nothing hidden, measure it honestly on a small subset, and keep the design open
to a more structured agent later.

```bash
make setup      # venv + deps + .env
$EDITOR .env    # paste your API key
make plan       # verify the whole topology — no model calls, no cost
make run        # agent -> patches -> harness -> report
```

**Contents** — [How it fits together](#how-it-fits-together) ·
[The two approaches](#the-two-approaches) · [What's in the repo](#whats-in-the-repo) ·
[Requirements](#requirements) · [Setup](#setup) · [Running it](#running-it) ·
[Configuration](#configuration) · [Outputs](#what-a-run-produces) ·
[Troubleshooting](#troubleshooting) · [Status](#status)

---

## How it fits together

Three things run, in two places:

```
  ┌─ your machine (native) ─────────────┐
  │                                     │
  │   gateway  127.0.0.1:8000 ──────────┼──── HTTPS ───►  free LLM API
  │   holds the API key                 │                 (Groq / Gemini /
  │                                     │                  OpenRouter)
  ├─ agent container (network_mode host)┤
  │                                     │
  │   runner ──► agent ──► bash tool    │
  │      │          ▲         │         │
  │      │          └─ obs ───┘         │
  │      ├──► evaluator                 │
  │      └──► telemetry / report        │
  └───────────┬─────────────────────────┘
              │ mounted /var/run/docker.sock
              ▼
     sibling containers on the HOST docker daemon
       ├─ per-task sandbox: the repo checked out at its base commit
       └─ SWE-bench harness: runs the hidden tests to score the patch
```

**Why split this way.** The gateway stays native and is the only process holding
the API key — the agent container never receives it, so nothing the model does
inside the sandbox can reach your credentials. The agent runs in a container so
its dependencies are pinned and reproducible, and it drives the host Docker
daemon through the mounted socket to spawn *sibling* sandboxes (one throwaway
container per task) rather than nesting Docker inside Docker.

Start the gateway **first** — the container's preflight fails fast without it.
`make run` does this for you.

---

## The two approaches

Both share the same skeleton — runner, sandbox, gateway, evaluator, telemetry —
and differ only in the agent's control loop.

| | **Approach A — `mini`** | **Approach B — `graph`** |
|---|---|---|
| Control flow | one `while` loop | explicit state machine (localize → edit → test → reflect) |
| Tools | a single bash tool | view / edit / search / run-tests / submit |
| Framework | none | LangGraph |
| Status | **implemented** | future extension |

Switching between them is **one line** in `config.yaml`:

```yaml
agent:
  type: mini     # -> graph
```

That is enforced by design: `runner.build_agent` resolves `agent.type` through a
registry using a lazy import, so the runner never names a concrete agent class.
Adding Approach B means adding two files (`src/agents/graph.py`,
`src/tools/structured.py`) and flipping that value — no existing file changes.

### How Approach A works

```
messages = system prompt + the issue text
loop until submit or budget exhausted:
    model returns ONE bash command in a ```bash block
    run it in the task sandbox (/testbed)
    append (command, truncated output) to history
patch = git diff against the instance's base commit
```

The model gets a raw shell and nothing else. It greps and cats to find the code,
edits with `sed`/heredocs, runs the project's own tests, and answers `submit`
when done. Everything it changed is collected as a `git diff` — no patch format
for it to get wrong.

---

## What's in the repo

| Path | What it does |
|---|---|
| `config.yaml` | every tunable: model, budgets, subset, eval settings. `agent.type` is the A/B switch |
| `.env` | your API key. Read **only** by the native gateway; never mounted into the container |
| `src/main.py` | entrypoint: preflight (gateway + docker), run plan, pipeline, eval, report |
| `src/config.py` | pydantic v2 schema + loader |
| `src/dataset.py` | `Instance` type; loads the SWE-bench subset (`limit` or pinned `instance_ids`) |
| `src/runner.py` | loops instances, builds the agent from `agent.type`, writes predictions |
| `src/history.py` | the conversation, trimmed to a token budget (oldest steps drop first) |
| `src/evaluator.py` | wraps `swebench.harness.run_evaluation` |
| `src/telemetry.py` | metrics, failure taxonomy, `report.md` / `report.json` |
| `src/llm/gateway_server.py` | the native gateway: `/health`, `/v1/chat/completions` |
| `src/llm/providers.py` | upstream adapters: groq, gemini, openrouter |
| `src/llm/client.py` | `GatewayClient` — the runner's only path to a model |
| `src/env/` | `Environment` interface + `DockerEnvironment` (per-task sandbox) |
| `src/tools/` | `Tool` interface + `BashTool` |
| `src/agents/` | `Agent` interface + `MiniAgent` |
| `docker/`, `docker-compose.yml`, `Makefile`, `run.sh` | build + run plumbing |
| `runs/<run_id>/` | outputs: predictions, trajectories, report, harness logs |

### The interfaces that hold it together

Five small contracts. Everything else is swappable behind them.

```python
Environment   reset(instance) · exec(cmd, timeout) · read_file · write_file · get_diff · close
Tool          name · spec · run(args, env) -> Observation
GatewayClient complete(messages) -> Completion
Agent         run(task) -> AgentResult          # patch + telemetry
History       append(action, observation) · messages() -> list[Message]
```

The runner depends on these and never on a concrete class.

---

## Requirements

| Requirement | Why | Check |
|---|---|---|
| Linux, x86_64 | SWE-bench images are x86_64; `network_mode: host` is Linux-only | `uname -m` |
| Docker, usable without sudo | sandboxes + the test harness are containers | `docker info` |
| [uv](https://docs.astral.sh/uv/) | host Python env | `uv --version` |
| Python 3.11 | pinned via `.python-version` | `uv python pin 3.11` |
| ≥16 GB RAM, ≥120 GB free disk | the SWE-bench harness, not the agent, sets this bar | `free -g`, `df -h` |
| A free-tier LLM API key | Groq, Gemini, or OpenRouter | — |

If `docker info` needs sudo: `sudo usermod -aG docker $USER`, then log out and
back in.

---

## Setup

```bash
make setup                 # uv venv + editable install + creates .env from the template
$EDITOR .env               # paste your key, e.g. GROQ_API_KEY=gsk_...
```

Pick your provider in `config.yaml` (`gateway.upstream`) and set `model.name` to
a model that provider serves.

| Provider | Where to get a key | Key looks like | `config.yaml` |
|---|---|---|---|
| **Groq** (default) | [console.groq.com/keys](https://console.groq.com/keys) | `gsk_…` | `upstream: groq`, `model.name: llama-3.3-70b-versatile` |
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `AIza…` or `AQ.Ab…` | `upstream: gemini`, `model.name: gemini-2.5-flash` |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `sk-or-v1-…` | `upstream: openrouter`, a `…:free` model |

**Heads-up on Gemini keys.** Google is migrating API keys from the `AIza…`
format to `AQ.Ab…` "auth keys". Some accounts can only issue `AQ.` keys, and
those are currently rejected by `generativelanguage.googleapis.com` on *every*
endpoint — native and OpenAI-compatible alike — with
`401 … Expected OAuth 2 access token`. No client-side change fixes it. If you hit
that, use Groq or OpenRouter; it is a one-line `upstream:` switch.

Verify the key before a real run:

```bash
make gateway
curl -s -X POST 127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"say ok"}]}'
```

A 200 with a `choices` array means you are good; 400/401 means the key or the
`upstream` setting is wrong. `./run.sh` performs this check automatically.

**The key lives only in `.env`, which only the native gateway process reads.** It
is never mounted into the agent container and never leaves your machine except in
the gateway's HTTPS call upstream.

---

## Running it

### Health-check the topology (no model calls, no cost)

```bash
make gateway     # start the native gateway in the background, wait for /health
make health      # -> {"status":"ok","upstream":"groq","model":"llama-3.3-70b-versatile"}
make plan        # build the image, run preflight in the container, print the run plan
make stop-gateway
```

Expected:

```
INFO main [ok]   gateway healthy at http://127.0.0.1:8000
INFO main [ok]   docker daemon 29.6.2

run plan
--------
  run dir      runs/20260720-183603
  dataset      swe-bench-lite [django__django-11099, ...]
  agent        mini  (A=mini, B=graph)
  model        llama-3.3-70b-versatile @ temp 0.0
  budgets      max_steps=40 max_cost_usd=0.0 step_timeout_s=120
  sandbox      task_timeout_s=900
  eval         run_eval=True workers=1 cache_level=env clean=True
```

This proves the three things that are easy to get wrong: the container can reach
the native gateway on `127.0.0.1`, it can drive the host Docker daemon through
the mounted socket, and the config parses.

### A full run

```bash
make run          # or ./run.sh, which checks prerequisites and the key first
```

Per instance it pulls the SWE-bench image, starts a sandbox container, runs the
agent loop, and collects `git diff` as the patch; then it scores every patch with
the SWE-bench harness and writes a report. Expect the first instance to be slow —
instance images are multi-GB.

It ends with the headline:

```
RESOLVED 1/3 (33.3%)
report written to runs/20260720-184922/report.md
```

### Re-scoring without re-running the agent

```bash
make eval      # re-run the harness over the last run's predictions.jsonl
make report    # rebuild report.md/report.json from an existing run
```

Both reuse the most recent directory under `runs/`; pass `--run-id <id>` to
target a specific one.

### All commands

```bash
make help          # list every target
make setup         # host venv + deps + .env
make gateway       # start the native gateway (background), wait for health
make health        # curl /health
make logs          # tail the gateway log (logs/gateway.log)
make stop-gateway  # stop it
make plan          # preflight + run plan, no model calls
make run           # gateway + full containerized pipeline
make eval          # re-run only the evaluator over existing predictions
make report        # rebuild the report from the last run
make clean-runs    # delete runs/
make clean         # docker system prune + delete runs/
```

---

## Configuration

Everything is in `config.yaml`. The settings that matter most on a 16 GB laptop:

```yaml
dataset:
  limit: 10               # start tiny (ignored when instance_ids is set)
  instance_ids:           # pin a few SAME-repo instances — biggest disk lever
    - django__django-11099

agent:
  type: mini              # THE A/B switch
  max_steps: 40           # hard stop on the agent loop
  step_timeout_s: 120     # hard stop on a single sandbox command
  history_token_budget: 60000

eval:
  max_workers: 1          # 16 GB RAM -> keep at 1
  cache_level: env        # reuse per-repo images, drop per-instance ones
  clean: true             # delete instance images after use
```

Environment images are one per repo and are the dominant disk cost, so pinning
`instance_ids` to a single repo is the most effective knob. Watch usage with
`docker system df` and reclaim with `make clean`.

Note the tradeoff of `cache_level: env` + `clean: true`: the harness deletes each
instance image after scoring it, so disk stays bounded but the next run re-pulls
them. That is the right default on a laptop. If you are iterating on one instance
repeatedly and have disk to spare, set `clean: false`.

---

## What a run produces

`runs/<run_id>/`, bind-mounted from the host so it survives the container:

| File | Contents |
|---|---|
| `predictions.jsonl` | one row per instance: `instance_id`, `model_patch` (SWE-bench format) |
| `results.json` | per-instance steps, tokens, exit status, wall clock |
| `trajectories/<id>.json` | every command the agent ran and its output |
| `report.md` / `report.json` | % resolved, per-instance table, failure taxonomy |
| `logs/` | the harness's own per-instance test logs |

Run directories are owned by root (the agent container runs as root to use the
docker socket). Use `make clean-runs` rather than `rm -rf` as your user.

The native gateway keeps its own state in **`runs/gateway/`** (stable across
runs, so the cache survives): `cache.json` (persisted response cache, keyed on
model+messages+temperature) and `calls.jsonl` (per-call log of tokens, cost,
cache hits, and retries). Its process log is separate, at `logs/gateway.log`.
`GET /stats` on the gateway returns live counters. A re-run of the same eval
replays from `cache.json` and costs nothing upstream.

### Reading the report

`report.md` leads with the resolve rate, then a per-instance table, then the
**failure taxonomy** — every unresolved instance in exactly one bucket:

| Bucket | Means | Usual fix |
|---|---|---|
| localization miss | the agent never edited anything | better search prompting |
| malformed patch | the harness could not apply/run the diff | stop it mangling files |
| tests still failing | patch applied, behaviour still wrong | the actual hard problem |
| out of steps | hit `max_steps` mid-task | raise the budget, or cut wasted steps |
| context overflow | history outgrew the budget | tune `history_token_budget` |
| API / rate-limit error | upstream refused | fix the key, lower the rate limits |
| sandbox error | the container never came up | check disk and image pulls |

Fix the biggest bucket, re-run the same subset, compare. That loop is the point
of the project.

---

## Troubleshooting

**`[FAIL] no gateway at http://127.0.0.1:8000`**
The gateway is not running or died at startup. `make gateway`, then check
`logs/gateway.log` — usually a missing key (it fails fast and names the empty env var).

**`[FAIL] docker info failed` inside the container**
Socket mount or permissions. Confirm `docker info` works on the host and that
`/var/run/docker.sock` is mounted (see `docker-compose.yml`). With rootless
Docker, match the socket's group instead of running the container as root.

**Upstream returns 400/401**
Wrong key type for the configured `upstream`, or a key the provider rejects. See
the provider table in [Setup](#setup).

**`runs/` is owned by root**
Expected — the agent container runs as root for socket access. Use
`make clean-runs`, or `sudo rm -rf runs/*`.

**Image pulls time out**
Instance images are multi-GB. Raise `sandbox.pull_timeout_s`.

**Rate-limit (429) errors from upstream**
The gateway paces requests under `gateway.requests_per_minute` /
`tokens_per_minute` and retries 429s with backoff, so these should be rare. If
you still see them, lower those two values to match your free tier — the token
limit (tpm) usually binds first.

---

## Status

Built in milestone order; each milestone had to meet its acceptance criteria
before the next one started.

| Milestone | What | Status |
|---|---|---|
| **M0** | Skeleton, config, native gateway (passthrough), container plumbing | ✅ done |
| **M1** | `DockerEnvironment`, `BashTool`, `MiniAgent`, `History`, predictions | ✅ done |
| **M2** | Dataset subset, SWE-bench harness wrapper, telemetry + report | ✅ done |
| **M3** | Gateway hardening: response cache, rpm/tpm rate limiting, 429 backoff, cost logging | ✅ done |
| **M4** | Improve Approach A: self-verification before submit, better localization — each behind a config flag | planned |
| **M5** | Approach B: `agents/graph.py` + `tools/structured.py`, selected by `agent.type: graph` | planned |


Conventions and invariants for contributors: [`CLAUDE.md`](CLAUDE.md).
