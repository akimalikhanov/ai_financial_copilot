#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

pids=()
cleanup() {
  kill "${pids[@]}" 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM

.venv/bin/uvicorn src.main:app --reload --host 0.0.0.0 &
pids+=($!)
.venv/bin/python -m src.workers.chat_worker &
pids+=($!)
(cd src/ui && npm run dev) &
pids+=($!)
wait
