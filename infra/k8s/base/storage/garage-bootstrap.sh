#!/bin/sh
# K8s rewrite of infra/scripts/garage_bootstrap.sh (stage-17 plan Phase 9, concept 3).
#
# Runs against Garage's Admin API (curl) rather than the CLI: dxflrs/garage:v2.2.0 is a scratch
# image containing nothing but the static /garage binary (no shell, no cp — confirmed via
# `docker export`), so the CLI-as-remote-client approach the plan originally recommended can't
# actually run a shell script from inside that image. curlimages/curl (already proven in
# es-bootstrap) has both curl and a shell, and the Admin API only needs the admin token, not the
# RPC secret + garage.toml the CLI approach would have required.
#
# Idempotent: safe to re-run (kubectl delete job garage-bootstrap && kubectl apply -k ...).
# Imports a pre-generated key pair instead of creating one at runtime (stage-17 plan Phase 9,
# concept 4) — the secret never needs to be read out of `kubectl logs`.

set -e

BASE="${GARAGE_ADMIN_URL:-http://garage:3903}"
AUTH="Authorization: Bearer ${GARAGE_ADMIN_TOKEN:?GARAGE_ADMIN_TOKEN not set}"
KEY_NAME="${GARAGE_KEY_NAME:-app-key}"
ZONE="${GARAGE_ZONE:-dc1}"
CAPACITY_BYTES="${GARAGE_CAPACITY_BYTES:-1000000000}"   # 1GB; declared metadata, not enforced

api_get()  { curl -fsS -H "$AUTH" "$BASE$1"; }
api_post() { curl -fsS -X POST -H "$AUTH" -H "Content-Type: application/json" -d "$2" "$BASE$1"; }

echo "Waiting for Garage admin API..."
i=0
until api_get /v2/GetClusterStatus >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -ge 30 ] && { echo "Garage did not become ready in time."; exit 1; }
  sleep 2
done

STATUS="$(api_get /v2/GetClusterStatus)"
NODE_ID="$(echo "$STATUS" | jq -r '.nodes[0].id')"
HAS_ROLE="$(echo "$STATUS" | jq -r '.nodes[0].role // empty')"

if [ -z "$NODE_ID" ] || [ "$NODE_ID" = "null" ]; then
  echo "Could not read node ID from GetClusterStatus."
  exit 1
fi

if [ -z "$HAS_ROLE" ]; then
  echo "Assigning layout for node $NODE_ID..."
  api_post /v2/UpdateClusterLayout \
    "{\"roles\":[{\"id\":\"$NODE_ID\",\"zone\":\"$ZONE\",\"capacity\":$CAPACITY_BYTES,\"tags\":[]}]}" >/dev/null
  CUR_VERSION="$(api_get /v2/GetClusterLayout | jq -r '.version')"
  api_post /v2/ApplyClusterLayout "{\"version\":$((CUR_VERSION + 1))}" >/dev/null
  echo "Layout applied."
else
  echo "Layout already assigned, skipping."
fi

BUCKETS_CSV="${GARAGE_BUCKETS:-pdfs,docling,rendered,chunks,pictures,langfuse}"
BUCKETS="$(printf '%s' "$BUCKETS_CSV" | tr ',' ' ')"

for b in $BUCKETS; do
  # Plain `jq` (not `-e`): -e sets jq's own exit code from the produced boolean, which under
  # `set -e` would kill the script on the very common, non-error case of "doesn't exist yet".
  EXISTS="$(api_get /v2/ListBuckets | jq --arg b "$b" 'any(.[]; .globalAliases[]? == $b)')"
  if [ "$EXISTS" = "true" ]; then
    echo "Bucket '$b' already exists, skipping."
  else
    api_post /v2/CreateBucket "{\"globalAlias\":\"$b\"}" >/dev/null
    echo "Bucket '$b' created."
  fi
done

# Key: imported from pre-generated material (make k8s-secrets), not created at runtime.
# ImportKey accepts a caller-supplied pair, so the value is already known (in
# overlays/kind/secrets/garage.env) before this Job runs — nothing to extract from a Job log.
: "${GARAGE_S3_ACCESS_KEY_ID:?GARAGE_S3_ACCESS_KEY_ID not set}"
: "${GARAGE_S3_SECRET_ACCESS_KEY:?GARAGE_S3_SECRET_ACCESS_KEY not set}"

KEY_EXISTS="$(api_get /v2/ListKeys | jq --arg n "$KEY_NAME" 'any(.[]; .name == $n)')"
if [ "$KEY_EXISTS" = "true" ]; then
  echo "Key '$KEY_NAME' already exists, skipping import."
else
  api_post /v2/ImportKey \
    "{\"accessKeyId\":\"$GARAGE_S3_ACCESS_KEY_ID\",\"secretAccessKey\":\"$GARAGE_S3_SECRET_ACCESS_KEY\",\"name\":\"$KEY_NAME\"}" >/dev/null
  echo "Key '$KEY_NAME' imported."
fi

# is expanded to:
# curl -fsS -X POST \
#   -H "Authorization: Bearer <token>" \
#   -H "Content-Type: application/json" \
#   -d '{"accessKeyId":"...","secretAccessKey":"...","name":"app-key"}' \
#   "http://garage:3903/v2/ImportKey"


# Allow key on buckets
BUCKETS_JSON="$(api_get /v2/ListBuckets)"
for b in $BUCKETS; do
  BUCKET_ID="$(echo "$BUCKETS_JSON" | jq -r --arg b "$b" '.[] | select(.globalAliases[]? == $b) | .id')"
  api_post /v2/AllowBucketKey \
    "{\"bucketId\":\"$BUCKET_ID\",\"accessKeyId\":\"$GARAGE_S3_ACCESS_KEY_ID\",\"permissions\":{\"read\":true,\"write\":true,\"owner\":true}}" >/dev/null
  echo "Key '$KEY_NAME' has RWO on bucket '$b'."
done

echo "Garage bootstrap complete."
