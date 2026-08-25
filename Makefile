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
	$(MAKE) k8s-check-es-bootstrap

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

# Local registry (docs/notes/gpu-on-local-k8s-review.md §9) replacing `kind load`, which
# re-imports the whole image on every rebuild. SHA tags make IfNotPresent correct by
# construction — but only if the tag actually changes on every rebuild. `git rev-parse HEAD`
# doesn't: uncommitted edits rebuild under the *same* tag, so kind's node-local containerd
# (IfNotPresent) keeps serving whatever it already cached for that tag and silently ignores
# the new content. GIT_SHA is therefore HEAD's SHA plus a hash of the working tree's actual
# diff (tracked modifications + untracked files) — any saved edit changes the tag, committed
# or not, so a rebuild is always a genuinely new image the node is forced to pull.
GIT_SHA ?= $(shell echo "$$(git rev-parse --short HEAD)-$$( \
	{ git diff HEAD -- . ':!infra/k8s/overlays/kind-deploy'; \
	  git ls-files -o --exclude-standard -z | xargs -0 -I{} cat {} 2>/dev/null; \
	} | sha256sum | cut -c1-8)")
REGISTRY := localhost:5001

.PHONY: k8s-registry
k8s-registry:
	@docker inspect kind-registry >/dev/null 2>&1 || \
	  docker run -d --restart=always -p "127.0.0.1:5001:5000" --name kind-registry registry:3
	@docker network inspect kind >/dev/null 2>&1 && \
	  { docker network connect kind kind-registry 2>/dev/null || true; } || \
	  echo "kind network not found yet — connect after k8s-up: docker network connect kind kind-registry"

.PHONY: docker-build-api
docker-build-api:
	docker build -f infra/docker/Dockerfile.api -t $(REGISTRY)/copilot/api:$(GIT_SHA) .

.PHONY: k8s-push-api
k8s-push-api:
	docker push $(REGISTRY)/copilot/api:$(GIT_SHA)

# NGINX_CONF_VARIANT=nginx.k8s.conf: plain proxy_pass (no Compose-only resolver trick,
# see src/ui/nginx.k8s.conf).
.PHONY: docker-build-frontend
docker-build-frontend:
	docker build -f infra/docker/Dockerfile.frontend \
	  --build-arg NGINX_CONF_VARIANT=nginx.k8s.conf \
	  -t $(REGISTRY)/copilot/frontend:$(GIT_SHA) .

.PHONY: k8s-push-frontend
k8s-push-frontend:
	docker push $(REGISTRY)/copilot/frontend:$(GIT_SHA)

# Shared image for both workers (Dockerfile.worker); they differ only by the `command:`
# in their Deployments, exactly as compose differentiates them by `command:`.
.PHONY: docker-build-worker
docker-build-worker:
	docker build -f infra/docker/Dockerfile.worker -t $(REGISTRY)/copilot/worker:$(GIT_SHA) .

.PHONY: k8s-push-worker
k8s-push-worker:
	docker push $(REGISTRY)/copilot/worker:$(GIT_SHA)

# Generated-only sibling of overlays/kind holding just the images: transformer — `kustomize
# edit set image` rewrites the whole file, which would destroy overlays/kind's hand comments.
# Never apply overlays/kind directly: it carries base's placeholder `copilot/*:dev` names,
# which resolve nowhere now that images live in the registry. K8S_OVERLAY is the only apply
# path, and every target that applies it depends on k8s-set-images to generate it first.
K8S_OVERLAY := infra/k8s/overlays/kind-deploy

.PHONY: k8s-set-images
k8s-set-images:
	@mkdir -p $(K8S_OVERLAY)
	@printf 'apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\nresources:\n  - ../kind\n' \
	  > $(K8S_OVERLAY)/kustomization.yaml
	cd $(K8S_OVERLAY) && kustomize edit set image \
	  copilot/api=$(REGISTRY)/copilot/api:$(GIT_SHA) \
	  copilot/frontend=$(REGISTRY)/copilot/frontend:$(GIT_SHA) \
	  copilot/worker=$(REGISTRY)/copilot/worker:$(GIT_SHA)

# Full build+push+deploy loop; use individual docker-build-*/k8s-push-* targets to iterate on one service.
#
# model-preload is deleted first because a Job's spec.template is immutable and its image tag
# changes on every rebuild — applying over a completed Job is a hard error that would fail the
# whole deploy. Recreating it is cheap: once the PVC holds the weights the Job is a no-op that
# exits in seconds, which also re-warms the cache if the PVC was ever wiped.
.PHONY: k8s-deploy
k8s-deploy: docker-build-api docker-build-frontend docker-build-worker
	$(MAKE) k8s-push-api k8s-push-frontend k8s-push-worker k8s-set-images
	kubectl delete job model-preload -n copilot --ignore-not-found
	kubectl apply -k $(K8S_OVERLAY)

# Builds the custom kind node image (NVIDIA toolkit + registry hosts.toml baked in). Re-run
# after editing the Dockerfile; tag must match kind-cluster.yaml's `image:`.
.PHONY: k8s-node-image
k8s-node-image:
	docker build -t copilot/kind-node:v1.36.1-nvidia1.18.1 \
	  -f infra/k8s/node-image/Dockerfile infra/k8s/node-image

# Creates the cluster with GPU access (custom node image + extraMounts) and the GPU device
# plugin. Host-side GPU prereqs are one-time per machine; see k8s-gpu-preflight below.
.PHONY: k8s-up
k8s-up: k8s-gpu-preflight
	kind create cluster --config infra/k8s/kind-cluster.yaml
	$(MAKE) k8s-registry
	$(MAKE) k8s-gpu-plugin
	$(MAKE) k8s-ingress-nginx

# Cluster-level add-on, not part of the app (make k8s-deploy's kustomize apply never touches
# it): installs once per cluster lifetime and survives every future k8s-deploy. Only needed
# again after kind delete cluster. kind-cluster.yaml already has the extraPortMappings
# (80/443) and ingress-ready=true node label this manifest's controller expects.
.PHONY: k8s-ingress-nginx
k8s-ingress-nginx:
	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
	kubectl wait --namespace ingress-nginx --for=condition=ready pod \
	  --selector=app.kubernetes.io/component=controller --timeout=120s

# Verifies the host-side GPU prerequisites before creating a cluster, so failures surface
# here with an actionable message instead of as a Pending pod an hour later.
.PHONY: k8s-gpu-preflight
k8s-gpu-preflight:
	@command -v nvidia-ctk >/dev/null || { echo "FAIL: nvidia-container-toolkit not installed"; exit 1; }
	@nvidia-smi -L >/dev/null 2>&1 || { echo "FAIL: nvidia-smi cannot see a GPU"; exit 1; }
	@docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' || \
	  { echo "FAIL: docker has no 'nvidia' runtime. Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"; exit 1; }
	@docker info 2>/dev/null | grep -q 'Default Runtime: nvidia' || \
	  { echo "FAIL: docker default runtime is not nvidia — kind creates its node container without a GPU otherwise."; \
	     echo "      Run: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker"; exit 1; }
	@grep -q '^[[:space:]]*accept-nvidia-visible-devices-as-volume-mounts[[:space:]]*=[[:space:]]*true' \
	  /etc/nvidia-container-runtime/config.toml || \
	  { echo "FAIL: accept-nvidia-visible-devices-as-volume-mounts is not true."; \
	     echo "      Run: sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place"; exit 1; }
	@# envvar-when-unprivileged is the self-grant bypass path and must be false.
	@grep -q '^[[:space:]]*accept-nvidia-visible-devices-envvar-when-unprivileged[[:space:]]*=[[:space:]]*false' \
	  /etc/nvidia-container-runtime/config.toml || \
	  { echo "FAIL: accept-nvidia-visible-devices-envvar-when-unprivileged is not false — any pod can self-grant the GPU."; \
	     echo "      Run: sudo nvidia-ctk config --set accept-nvidia-visible-devices-envvar-when-unprivileged=false --in-place && sudo systemctl restart docker"; exit 1; }
	@echo "GPU preflight OK"

# Applied outside the kind overlay: it targets kube-system, and the overlay's
# `namespace: copilot` + commonLabels would rewrite its namespace and mutate the
# DaemonSet's immutable selector.
.PHONY: k8s-gpu-plugin
k8s-gpu-plugin:
	kubectl apply -f infra/k8s/overlays/kind/gpu/device-plugin.yaml
	@echo "waiting for nvidia.com/gpu to appear on the node..."
	@for i in $$(seq 1 60); do \
	  if [ -n "$$(kubectl get node copilot-control-plane -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null)" ]; then \
	    echo "GPU allocatable: $$(kubectl get node copilot-control-plane -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"; exit 0; fi; \
	  sleep 2; done; \
	  echo "TIMEOUT: nvidia.com/gpu never became allocatable. kubectl -n kube-system logs -l name=nvidia-device-plugin-ds"; exit 1

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

# infra/k8s/base/search/es-bootstrap/*.{sh,json} are copies of infra/docker/elasticsearch/*.
# infra/docker/elasticsearch/ is the source of truth; run this after editing any of them.
.PHONY: k8s-sync-es-bootstrap
k8s-sync-es-bootstrap:
	cp infra/docker/elasticsearch/bootstrap.sh infra/k8s/base/search/es-bootstrap/bootstrap.sh
	cp infra/docker/elasticsearch/ilm-policy.json infra/k8s/base/search/es-bootstrap/ilm-policy.json
	cp infra/docker/elasticsearch/index-template.json infra/k8s/base/search/es-bootstrap/index-template.json

# Fails if the K8s es-bootstrap copies have drifted from infra/docker/elasticsearch/. Run
# `make k8s-sync-es-bootstrap` to fix. Wired into `lint` so CI catches silent drift.
.PHONY: k8s-check-es-bootstrap
k8s-check-es-bootstrap:
	@diff -q infra/docker/elasticsearch/bootstrap.sh infra/k8s/base/search/es-bootstrap/bootstrap.sh
	@diff -q infra/docker/elasticsearch/ilm-policy.json infra/k8s/base/search/es-bootstrap/ilm-policy.json
	@diff -q infra/docker/elasticsearch/index-template.json infra/k8s/base/search/es-bootstrap/index-template.json

# Jobs are immutable (spec.template can't change in place); delete-then-apply is the explicit
# re-run path recommended in Phase 8, mirroring how `docker compose run --rm garage-bootstrap`
# is already a manual step.
#
# Depends on k8s-set-images so this applies the same registry-tagged overlay k8s-deploy does.
# Applying overlays/kind directly here would revert every app image to base's placeholder
# `copilot/*:dev`, which no longer exists on the node or in the registry -> ErrImagePull.
.PHONY: k8s-bootstrap
k8s-bootstrap: k8s-set-images
	kubectl delete job es-bootstrap garage-bootstrap -n copilot --ignore-not-found
	kubectl apply -k $(K8S_OVERLAY)

# Warms the hf-model-cache PVC with Docling's picture-description VLM. Without it, the first
# PDF upload after a fresh cluster stalls inside the Celery task on a multi-GB HuggingFace
# download, which the UI reports as a failed ingest. Run once per cluster lifetime (the PVC
# survives pod restarts); re-runs are cheap no-ops once the snapshot is cached.
#
# Same delete-then-apply shape as k8s-bootstrap: Jobs are immutable, and k8s-set-images is
# required so the Job gets the registry-tagged worker image rather than base's `copilot/*:dev`.
.PHONY: k8s-preload-models
k8s-preload-models: k8s-set-images
	kubectl delete job model-preload -n copilot --ignore-not-found
	kubectl apply -k $(K8S_OVERLAY)
	kubectl wait --namespace copilot --for=condition=complete job/model-preload --timeout=3600s

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
	@test -f $(K8S_SECRETS)/garage.env || { \
	  { echo "garage_rpc_secret=$$(openssl rand -hex 32)"; \
	    echo "garage_admin_token=$$(openssl rand -base64 32)"; \
	    echo "garage_metrics_token=$$(openssl rand -base64 32)"; \
	    echo "garage_s3_access_key_id=GK$$(openssl rand -hex 12)"; \
	    echo "garage_s3_secret_access_key=$$(openssl rand -hex 32)"; \
	  } > $(K8S_SECRETS)/garage.env; chmod 600 $(K8S_SECRETS)/garage.env; \
	  echo "generated $(K8S_SECRETS)/garage.env"; }
	@# The S3 key pair above is pre-generated and imported by the garage-bootstrap Job (stage-17
	@# plan Phase 9, concept 4) rather than created at runtime, so it's already known here — write
	@# it straight into app.env instead of a post-bootstrap sync step. app.env gains more keys
	@# (JWT_SECRET, LLM provider keys, ...) once the app tier lands; this only owns the AWS_* pair.
	@test -f $(K8S_SECRETS)/app.env || touch $(K8S_SECRETS)/app.env && chmod 600 $(K8S_SECRETS)/app.env
	@grep -q '^AWS_ACCESS_KEY_ID=' $(K8S_SECRETS)/app.env || { \
	  ACCESS_KEY="$$(grep '^garage_s3_access_key_id=' $(K8S_SECRETS)/garage.env | cut -d= -f2)"; \
	  SECRET_KEY="$$(grep '^garage_s3_secret_access_key=' $(K8S_SECRETS)/garage.env | cut -d= -f2)"; \
	  { echo "AWS_ACCESS_KEY_ID=$$ACCESS_KEY"; \
	    echo "AWS_SECRET_ACCESS_KEY=$$SECRET_KEY"; \
	  } >> $(K8S_SECRETS)/app.env; \
	  echo "wrote AWS_* creds into $(K8S_SECRETS)/app.env"; }
	@# Phase 10 (app tier): JWT_SECRET is generated fresh; APP_DB_PASSWORD/REDIS_PASSWORD are
	@# copied from the postgres/redis secret files generated above (single source of truth,
	@# api container just needs its own secretRef with the same values). LLM provider keys and
	@# the Langfuse keypair are external/not-yet-generated — left blank for manual entry.
	@grep -q '^JWT_SECRET=' $(K8S_SECRETS)/app.env || { \
	  APP_DB_PASSWORD="$$(grep '^APP_DB_PASSWORD=' $(K8S_SECRETS)/postgres.env | cut -d= -f2)"; \
	  REDIS_PASSWORD="$$(grep '^REDIS_PASSWORD=' $(K8S_SECRETS)/redis.env | cut -d= -f2)"; \
	  { echo "JWT_SECRET=$$(openssl rand -base64 32 | tr -d '/+=')"; \
	    echo "OPENAI_API_KEY="; \
	    echo "GOOGLE_API_KEY="; \
	    echo "HF_TOKEN="; \
	    echo "APP_DB_USER=app"; \
	    echo "APP_DB_PASSWORD=$$APP_DB_PASSWORD"; \
	    echo "REDIS_PASSWORD=$$REDIS_PASSWORD"; \
	    echo "LANGFUSE_PUBLIC_KEY="; \
	    echo "LANGFUSE_SECRET_KEY="; \
	  } >> $(K8S_SECRETS)/app.env; \
	  echo "wrote JWT_SECRET/APP_DB_*/REDIS_PASSWORD into $(K8S_SECRETS)/app.env (fill in LLM/Langfuse keys manually)"; }
	@test -f $(K8S_SECRETS)/grafana.env || { \
	  { echo "GRAFANA_USER=admin"; \
	    echo "GRAFANA_PASS=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	  } > $(K8S_SECRETS)/grafana.env; chmod 600 $(K8S_SECRETS)/grafana.env; \
	  echo "generated $(K8S_SECRETS)/grafana.env"; }
	@# Phase 16: ClickHouse credentials. Read by the ClickHouse StatefulSet *and* by both
	@# Langfuse pods from this same Secret, so the two can never drift apart.
	@test -f $(K8S_SECRETS)/clickhouse.env || { \
	  { echo "CLICKHOUSE_USER=clickhouse"; \
	    echo "CLICKHOUSE_PASSWORD=$$(openssl rand -base64 32 | tr -d '/+=')"; \
	  } > $(K8S_SECRETS)/clickhouse.env; chmod 600 $(K8S_SECRETS)/clickhouse.env; \
	  echo "generated $(K8S_SECRETS)/clickhouse.env"; }
	@# Phase 16: Langfuse. Mirrors infra/scripts/langfuse_bootstrap.sh. Connection URLs are
	@# composed here (rather than assembled in the pod) because Langfuse wants single DSN
	@# strings and K8s env can't interpolate one Secret key into another.
	@test -f $(K8S_SECRETS)/langfuse.env || { \
	  LANGFUSE_DB_PASSWORD="$$(grep '^LANGFUSE_DB_PASSWORD=' $(K8S_SECRETS)/postgres.env | cut -d= -f2)"; \
	  REDIS_PASSWORD="$$(grep '^REDIS_PASSWORD=' $(K8S_SECRETS)/redis.env | cut -d= -f2)"; \
	  CH_USER="$$(grep '^CLICKHOUSE_USER=' $(K8S_SECRETS)/clickhouse.env | cut -d= -f2)"; \
	  CH_PASS="$$(grep '^CLICKHOUSE_PASSWORD=' $(K8S_SECRETS)/clickhouse.env | cut -d= -f2)"; \
	  S3_KEY="$$(grep '^garage_s3_access_key_id=' $(K8S_SECRETS)/garage.env | cut -d= -f2)"; \
	  S3_SECRET="$$(grep '^garage_s3_secret_access_key=' $(K8S_SECRETS)/garage.env | cut -d= -f2)"; \
	  PK="pk-lf-$$(cat /proc/sys/kernel/random/uuid)"; \
	  SK="sk-lf-$$(cat /proc/sys/kernel/random/uuid)"; \
	  { echo "DATABASE_URL=postgresql://langfuse:$$LANGFUSE_DB_PASSWORD@postgres:5432/langfuse"; \
	    echo "CLICKHOUSE_MIGRATION_URL=clickhouse://$$CH_USER:$$CH_PASS@clickhouse:9000"; \
	    echo "REDIS_AUTH=$$REDIS_PASSWORD"; \
	    echo "NEXTAUTH_SECRET=$$(openssl rand -base64 32)"; \
	    echo "SALT=$$(openssl rand -base64 24)"; \
	    echo "ENCRYPTION_KEY=$$(openssl rand -hex 32)"; \
	    for scope in EVENT_UPLOAD MEDIA_UPLOAD BATCH_EXPORT; do \
	      echo "LANGFUSE_S3_$${scope}_ACCESS_KEY_ID=$$S3_KEY"; \
	      echo "LANGFUSE_S3_$${scope}_SECRET_ACCESS_KEY=$$S3_SECRET"; \
	    done; \
	    echo "LANGFUSE_INIT_ORG_ID=$$(cat /proc/sys/kernel/random/uuid)"; \
	    echo "LANGFUSE_INIT_ORG_NAME=AI Financial Copilot"; \
	    echo "LANGFUSE_INIT_PROJECT_ID=$$(cat /proc/sys/kernel/random/uuid)"; \
	    echo "LANGFUSE_INIT_PROJECT_NAME=copilot"; \
	    echo "LANGFUSE_INIT_PROJECT_PUBLIC_KEY=$$PK"; \
	    echo "LANGFUSE_INIT_PROJECT_SECRET_KEY=$$SK"; \
	    echo "LANGFUSE_INIT_USER_EMAIL=admin@copilot.local"; \
	    echo "LANGFUSE_INIT_USER_NAME=admin"; \
	    echo "LANGFUSE_INIT_USER_PASSWORD=$$(openssl rand -base64 16 | tr -d '/+=')"; \
	  } > $(K8S_SECRETS)/langfuse.env; chmod 600 $(K8S_SECRETS)/langfuse.env; \
	  echo "generated $(K8S_SECRETS)/langfuse.env"; }
	@# The app tier authenticates to Langfuse with the seeded project's key pair, so the two
	@# blank placeholders written into app.env in Phase 10 are filled from langfuse.env here.
	@grep -q '^LANGFUSE_PUBLIC_KEY=.' $(K8S_SECRETS)/app.env || { \
	  PK="$$(grep '^LANGFUSE_INIT_PROJECT_PUBLIC_KEY=' $(K8S_SECRETS)/langfuse.env | cut -d= -f2)"; \
	  SK="$$(grep '^LANGFUSE_INIT_PROJECT_SECRET_KEY=' $(K8S_SECRETS)/langfuse.env | cut -d= -f2)"; \
	  sed -i "s|^LANGFUSE_PUBLIC_KEY=.*|LANGFUSE_PUBLIC_KEY=$$PK|; s|^LANGFUSE_SECRET_KEY=.*|LANGFUSE_SECRET_KEY=$$SK|" \
	    $(K8S_SECRETS)/app.env; \
	  echo "wrote LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY into $(K8S_SECRETS)/app.env"; }
