#!/usr/bin/env bash
# Run API + chat worker + ingestion worker + UI in one terminal.
# On Ctrl+C, tears down the *entire* process tree of each child — Celery
# MainProcesses fork a pool of workers that are NOT direct children of this
# script, so killing only $! leaks them (they get reparented to init and keep
# draining the queue). We start each child in its own process group and signal
# the whole group.
cd "$(dirname "$0")/.."

# Each child leads its own process group (setsid) so we can kill the group.
pgids=()
start() {
  setsid "$@" &
  # The child is its own session/group leader; its PGID == its PID.
  pgids+=($!)
}

cleanup() {
  trap '' SIGINT SIGTERM  # ignore further signals while we tear down
  echo
  echo "run_dev: shutting down workers…"
  # Negative PID == signal the entire process group (MainProcess + pool).
  for pgid in "${pgids[@]}"; do
    kill -TERM "-$pgid" 2>/dev/null
  done
  # Give Celery a moment for warm shutdown, then hard-kill any stragglers.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    still_running=0
    for pgid in "${pgids[@]}"; do
      kill -0 "-$pgid" 2>/dev/null && still_running=1
    done
    [ "$still_running" -eq 0 ] && break
    sleep 0.5
  done
  for pgid in "${pgids[@]}"; do
    kill -KILL "-$pgid" 2>/dev/null
  done
  exit 0
}
trap cleanup SIGINT SIGTERM

export PYTORCH_ALLOC_CONF=expandable_segments:True

start .venv/bin/uvicorn src.main:app --reload --reload-dir src --host 0.0.0.0
start .venv/bin/python -m src.workers.chat_worker
start .venv/bin/python -m src.workers.ingestion_worker
start bash -c 'cd src/ui && exec npm run dev'

# Wait for all children. Note: no `set -e` — a single worker crashing must NOT
# abort the script, or `wait` is skipped and survivors get orphaned.
wait
