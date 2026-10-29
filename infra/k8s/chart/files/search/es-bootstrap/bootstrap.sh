#!/bin/sh
set -e

ES="${ES_URL:-http://elasticsearch:9200}"

echo "Waiting for Elasticsearch..."
until curl -fsS "${ES}/_cluster/health" >/dev/null 2>&1; do
  sleep 2
done

echo "Applying ILM policy..."
curl -fsSX PUT "${ES}/_ilm/policy/copilot-logs-ilm" \
  -H "Content-Type: application/json" \
  -d @/bootstrap/ilm-policy.json

echo "Applying index template..."
curl -fsSX PUT "${ES}/_index_template/copilot-logs" \
  -H "Content-Type: application/json" \
  -d @/bootstrap/index-template.json

# `copilot-logs` must resolve to the rollover alias, never to a concrete index of that name.
# _cat/indices returns the backing index (copilot-logs-000001) for the alias, but the name
# itself when a writer raced this job and auto-created it. Report that instead of letting the
# PUT below fail with a bare 400.
resolved=$(curl -fsS "${ES}/_cat/indices/copilot-logs?h=index" 2>/dev/null | tr -d ' \r\n')
if [ "$resolved" = "copilot-logs" ]; then
  echo "FATAL: 'copilot-logs' exists as a concrete index, so the rollover alias cannot be created." >&2
  echo "       A log writer auto-created it before this job ran. Its contents are disposable" >&2
  echo "       logs; delete it and re-run this job:" >&2
  echo "         curl -XDELETE '${ES}/copilot-logs'" >&2
  exit 1
fi

# Create the initial write index (only if alias doesn't exist yet)
if ! curl -fsS "${ES}/_alias/copilot-logs" >/dev/null 2>&1; then
  echo "Creating initial write index..."
  curl -fsSX PUT "${ES}/copilot-logs-000001" \
    -H "Content-Type: application/json" \
    -d '{"aliases":{"copilot-logs":{"is_write_index":true}}}'
fi

echo "Elasticsearch bootstrap complete."
