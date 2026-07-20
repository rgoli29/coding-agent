#!/usr/bin/env bash
# One-command run: check prerequisites, start the native gateway, run the
# containerized pipeline. Equivalent to `make setup` (first time) + `make run`,
# but with clearer diagnostics. See README.md for the full guide.
set -euo pipefail

cd "$(dirname "$0")"

die() { echo "error: $*" >&2; exit 1; }

echo "==> checking prerequisites"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (add yourself to the docker group?)"
command -v uv >/dev/null || die "uv is not installed — https://docs.astral.sh/uv/"

if [ ! -d .venv ]; then
  echo "==> creating the host venv"
  uv venv
  uv pip install -e .
fi

if [ ! -f .env ]; then
  make --no-print-directory .env
  die "put your API key in .env, then re-run. See README.md (Setup)."
fi

grep -qE '^[A-Z_]+_API_KEY=.+' .env || die "no API key set in .env. See README.md (Setup)."

echo "==> starting the native gateway"
make gateway

echo "==> checking the key against the upstream provider"
resp=$(curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"say ok"}]}')
if ! grep -q '"choices"' <<<"$resp"; then
  echo "$resp" | head -c 300 >&2
  echo >&2
  die "the gateway could not complete a test call — check your key and gateway.upstream"
fi
echo "    upstream ok"

echo "==> running the pipeline"
docker compose up --build agent

echo
echo "==> done. Latest report:"
ls -1dt runs/*/report.md 2>/dev/null | head -1 || echo "    (no report written)"
