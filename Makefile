# Run backend + worker + frontend (single terminal)
.PHONY: dev
dev:
	./scripts/run_dev.sh

# Run only the API (uvicorn)
.PHONY: api
api:
	.venv/bin/uvicorn src.main:app --reload --host 0.0.0.0

# Run only the chat worker (Celery worker, queue: chat)
# PROMETHEUS_MULTIPROC_DIR lets prefork children share metrics with the parent's
# :9100 metrics server. Must be set before Python imports prometheus_client.
.PHONY: worker
worker:
	rm -rf /tmp/prom_chat && mkdir -p /tmp/prom_chat && \
	PROMETHEUS_MULTIPROC_DIR=/tmp/prom_chat .venv/bin/python -m src.workers.chat_worker

# Run only the ingestion (Celery) worker
.PHONY: worker-ingestion
worker-ingestion:
	rm -rf /tmp/prom_ingestion && mkdir -p /tmp/prom_ingestion && \
	PROMETHEUS_MULTIPROC_DIR=/tmp/prom_ingestion .venv/bin/python -m src.workers.ingestion_worker

# Run only the frontend
.PHONY: ui
ui:
	cd src/ui && npm run dev

.PHONY: lint
lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	$(MAKE) k8s-check-initdb

.PHONY: typecheck
typecheck:
	.venv/bin/pyright --level error

.PHONY: test-unit
test-unit:
	.venv/bin/python -m pytest tests/unit/

.PHONY: test-integration
test-integration:
	.venv/bin/python -m pytest -m integration tests/integration/

.PHONY: test
test: test-unit test-integration

.PHONY: test-cov
test-cov:
	.venv/bin/python -m pytest tests/unit/ --cov=src --cov-report=term --cov-report=html

# Rebuild + restart the containerized stack, then drop the dangling images
# left behind by the previous build (same tag, now untagged).
.PHONY: docker-rebuild
docker-rebuild:
	cd infra/docker && docker compose --env-file ../../.env up -d --build
	docker image prune -f

.PHONY: k8s-up
k8s-up:
	kind create cluster --config infra/k8s/kind-cluster.yaml

.PHONY: k8s-down
k8s-down:
	kind delete cluster --name copilot

# infra/k8s/base/data/initdb/*.sh and infra/k8s/base/data/pgbouncer-entrypoint.sh are copies
# of infra/scripts/db_init/*.sh and infra/scripts/pgbouncer-entrypoint.sh — kept under the
# kustomization root so configMapGenerator never needs --load-restrictor LoadRestrictionsNone.
# infra/scripts/ is the source of truth; run this after editing any of them.
.PHONY: k8s-sync-initdb
k8s-sync-initdb:
	cp infra/scripts/db_init/00_db_init.sh infra/k8s/base/data/initdb/00_db_init.sh
	cp infra/scripts/db_init/01_create_app_tables.sh infra/k8s/base/data/initdb/01_create_app_tables.sh
	cp infra/scripts/pgbouncer-entrypoint.sh infra/k8s/base/data/pgbouncer-entrypoint.sh

# Fails if the K8s initdb/pgbouncer-entrypoint copies have drifted from infra/scripts/. Run
# `make k8s-sync-initdb` to fix. Wired into `lint` so CI catches silent drift.
.PHONY: k8s-check-initdb
k8s-check-initdb:
	@diff -q infra/scripts/db_init/00_db_init.sh infra/k8s/base/data/initdb/00_db_init.sh
	@diff -q infra/scripts/db_init/01_create_app_tables.sh infra/k8s/base/data/initdb/01_create_app_tables.sh
	@diff -q infra/scripts/pgbouncer-entrypoint.sh infra/k8s/base/data/pgbouncer-entrypoint.sh

K8S_SECRETS := infra/k8s/overlays/kind/secrets

.PHONY: k8s-secrets
k8s-secrets:
	@mkdir -p $(K8S_SECRETS)
	@test -f $(K8S_SECRETS)/postgres.env || { \
	  { echo "POSTGRES_USER=postgres"; \
	    echo "POSTGRES_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	    echo "APP_DB_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	    echo "LANGFUSE_DB_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	  } > $(K8S_SECRETS)/postgres.env; chmod 600 $(K8S_SECRETS)/postgres.env; \
	  echo "generated $(K8S_SECRETS)/postgres.env"; }
	@test -f $(K8S_SECRETS)/redis.env || { \
	  { echo "REDIS_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	  } > $(K8S_SECRETS)/redis.env; chmod 600 $(K8S_SECRETS)/redis.env; \
	  echo "generated $(K8S_SECRETS)/redis.env"; }
	@# ... one such block per secret file; extend in Phases 9, 15, 16
