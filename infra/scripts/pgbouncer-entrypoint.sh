#!/bin/sh
set -eu

: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${APP_DB:?APP_DB is required}"
: "${APP_DB_USER:?APP_DB_USER is required}"
: "${APP_DB_PASSWORD:?APP_DB_PASSWORD is required}"
: "${LANGFUSE_DB:?LANGFUSE_DB is required}"
: "${LANGFUSE_DB_USER:?LANGFUSE_DB_USER is required}"
: "${LANGFUSE_DB_PASSWORD:?LANGFUSE_DB_PASSWORD is required}"

PGB_POSTGRES_HOST="${PGB_POSTGRES_HOST:-postgres}"
PGB_POSTGRES_PORT="${PGB_POSTGRES_PORT:-5432}"
PGB_LISTEN_ADDR="${PGB_LISTEN_ADDR:-0.0.0.0}"
PGB_LISTEN_PORT="${PGB_LISTEN_PORT:-6432}"

mkdir -p /etc/pgbouncer

# PgBouncer needs a userlist entry for the auth_user (here: POSTGRES_USER).
# We do NOT store app/langfuse user passwords here — auth_query will fetch hashes from Postgres.
# Note: MD5 is deprecated but still works with PostgreSQL 16. For production, consider migrating to scram-sha-256.
cat > /etc/pgbouncer/userlist.txt <<EOF
"${POSTGRES_USER}" "${POSTGRES_PASSWORD}"
"${APP_DB_USER}" "${APP_DB_PASSWORD}"
"${LANGFUSE_DB_USER}" "${LANGFUSE_DB_PASSWORD}"
EOF

cat > /etc/pgbouncer/pgbouncer.ini <<EOF
[databases]
# route specific DB names to Postgres
${POSTGRES_DB} = host=${PGB_POSTGRES_HOST} port=${PGB_POSTGRES_PORT} dbname=${POSTGRES_DB} password=${POSTGRES_PASSWORD}
${APP_DB}      = host=${PGB_POSTGRES_HOST} port=${PGB_POSTGRES_PORT} dbname=${APP_DB} password=${APP_DB_PASSWORD}
${LANGFUSE_DB} = host=${PGB_POSTGRES_HOST} port=${PGB_POSTGRES_PORT} dbname=${LANGFUSE_DB} password=${LANGFUSE_DB_PASSWORD}

[pgbouncer]
listen_addr = ${PGB_LISTEN_ADDR}
listen_port = ${PGB_LISTEN_PORT}

auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

pool_mode = transaction
default_pool_size = 30
min_pool_size = 0
max_client_conn = 200

server_reset_query = DISCARD ALL
ignore_startup_parameters = extra_float_digits

log_connections = 1
log_disconnections = 1
EOF

echo "[pgbouncer] config written. starting pgbouncer..."
exec pgbouncer /etc/pgbouncer/pgbouncer.ini
