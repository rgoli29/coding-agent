#!/usr/bin/env bash
# Poll the native gateway's /health until it answers 200 (or we give up).
set -euo pipefail

URL="${1:-http://127.0.0.1:8000/health}"
ATTEMPTS="${2:-40}"

for i in $(seq 1 "$ATTEMPTS"); do
  if curl -fsS --max-time 2 "$URL" >/dev/null 2>&1; then
    echo "gateway healthy at $URL"
    exit 0
  fi
  sleep 0.5
done

echo "gateway did not become healthy at $URL after $ATTEMPTS attempts" >&2
echo "check the log:  tail .gateway.log" >&2
exit 1
