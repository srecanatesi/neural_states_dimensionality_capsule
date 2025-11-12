#!/usr/bin/env bash
set -euo pipefail

echo "SESSION_ID=${SESSION_ID:-}" | tee /results/logs/env_${SESSION_ID:-unknown}.log || true

python /code/process_session.py "$SESSION_ID" | tee \
  "/results/logs/session_${SESSION_ID:-unknown}.log"




