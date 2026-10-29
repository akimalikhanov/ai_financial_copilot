#!/bin/sh
set -eu

echo "[init] creating users and databases…"

# Require vars (fail fast with clear message)
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${LANGFUSE_DB:?LANGFUSE_DB is required}"
: "${LANGFUSE_DB_USER:?LANGFUSE_DB_USER is required}"
: "${LANGFUSE_DB_PASSWORD:?LANGFUSE_DB_PASSWORD is required}"
: "${APP_DB:?APP_DB is required}"
: "${APP_DB_USER:?APP_DB_USER is required}"
: "${APP_DB_PASSWORD:?APP_DB_PASSWORD is required}"

# 1) create users if they don't exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${LANGFUSE_DB_USER}') THEN
      CREATE USER ${LANGFUSE_DB_USER} WITH PASSWORD '${LANGFUSE_DB_PASSWORD}';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${APP_DB_USER}') THEN
      CREATE USER ${APP_DB_USER} WITH PASSWORD '${APP_DB_PASSWORD}';
    END IF;
  END
  \$\$;
EOSQL

# 2) create databases if they don't exist (CREATE DATABASE can't run in DO)
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
  SELECT 'CREATE DATABASE ${LANGFUSE_DB} OWNER ${LANGFUSE_DB_USER}'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${LANGFUSE_DB}')\\gexec

  SELECT 'CREATE DATABASE ${APP_DB} OWNER ${APP_DB_USER}'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${APP_DB}')\\gexec
EOSQL

echo "[init] done."
