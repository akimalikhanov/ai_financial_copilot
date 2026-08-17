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

# Create the initial write index (only if alias doesn't exist yet)
if ! curl -fsS "${ES}/_alias/copilot-logs" >/dev/null 2>&1; then
  echo "Creating initial write index..."
  curl -fsSX PUT "${ES}/copilot-logs-000001" \
    -H "Content-Type: application/json" \
    -d '{"aliases":{"copilot-logs":{"is_write_index":true}}}'
fi

echo "Elasticsearch bootstrap complete."
