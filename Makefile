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
	$(MAKE) k8s-check-dashboards
	$(MAKE) k8s-check-loadtest

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

# Load test against the docker-compose stack (mode A/B, see CLAUDE.md), headless with a
# text summary on stdout. USERS/SPAWN_RATE/DURATION are overridable: `make loadtest-local
# USERS=40 DURATION=30m`. Not a CI gate — see docs/notes/loadtest-concepts.md.
#
# Point the api/worker-chat env at infra/config/models.loadtest.yaml first (MODELS_CONFIG_PATH)
# unless you mean to spend real LLM money — see docs/notes/loadtest-concepts.md §6.
LOADTEST_HOST := http://localhost:$(or $(API_PORT),8000)
USERS ?= 5
SPAWN_RATE ?= 1
DURATION ?= 5m

.PHONY: loadtest-local
loadtest-local:
	.venv/bin/locust -f infra/loadtest/locustfile.py --host $(LOADTEST_HOST) \
	  --headless -u $(USERS) -r $(SPAWN_RATE) -t $(DURATION)

# Same target with the interactive web UI instead of a headless run — open the URL Locust
# prints (default http://localhost:8089) to start/stop the run and watch charts live.
.PHONY: loadtest-local-ui
loadtest-local-ui:
	.venv/bin/locust -f infra/loadtest/locustfile.py --host $(LOADTEST_HOST)

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
# Every Job the overlay applies is deleted first, because a Job's spec.template is immutable:
# applying over an existing Job is a hard error that fails the whole deploy. Each of the three
# has a template that changes on its own schedule — model-preload's image tag on every rebuild,
# and the two bootstraps' configMapGenerator hash suffix whenever their scripts change — so any
# of them can wedge a deploy. Recreating all three is cheap: each is a no-op that exits in
# seconds once its PVC/bucket/index already exists.
#
# Leaving one out is not merely a failed deploy. A Job created against a since-renamed
# ConfigMap can never be repaired by apply, and its pod sits in ContainerCreating on
# FailedMount indefinitely — observed on es-bootstrap and garage-bootstrap for 22h.
.PHONY: k8s-deploy
k8s-deploy: docker-build-api docker-build-frontend docker-build-worker
	$(MAKE) k8s-push-api k8s-push-frontend k8s-push-worker k8s-set-images
	kubectl delete job model-preload es-bootstrap garage-bootstrap -n copilot --ignore-not-found
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

# infra/k8s/base/observability/grafana/dashboards/* are copies of infra/docker/grafana/dashboards/*.
# infra/docker/ is the source of truth. worker-health.json is excluded on purpose: the K8s copy
# carries extra GPU panels (dcgm-exporter) that have no compose equivalent.
.PHONY: k8s-sync-dashboards
k8s-sync-dashboards:
	@for d in api-overview agentic-rag cost-tokens logs-explorer eval-canary; do \
	  cp infra/docker/grafana/dashboards/$$d.json infra/k8s/base/observability/grafana/dashboards/$$d.json; \
	done

# Fails if the K8s dashboard copies have drifted. Run `make k8s-sync-dashboards` to fix.
.PHONY: k8s-check-dashboards
k8s-check-dashboards:
	@for d in api-overview agentic-rag cost-tokens logs-explorer eval-canary; do \
	  diff -q infra/docker/grafana/dashboards/$$d.json infra/k8s/base/observability/grafana/dashboards/$$d.json || exit 1; \
	done

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

# infra/k8s/loadtest/{locustfile.py,models.yaml} are copies of infra/loadtest/locustfile.py
# and infra/config/models.loadtest.yaml. Kustomize refuses to read files outside its own
# directory, so the copies exist for configMapGenerator; the originals stay the source of truth.
# Builds the three ingestion fixture classes into the hostPath the fixtures PVC is backed by
# (docs/notes/loadtest-readiness-audit.md §8 items 1.2-1.3). SOURCE must be a real filing:
# T11 measures service time, and a synthetic PDF parses in seconds where a real one takes
# minutes. The generated PDFs are gitignored and reproducible from this command.
#   make loadtest-fixtures SOURCE="data/corpus/pdfs/Microsoft Corporation.pdf"
# 0 = the whole filing, for both. Neither capacity fixture may be truncated: the corpus runs
# 11-1043 pages (median 128), so the 10-page versions these used to build were shorter than the
# smallest real document in it, and sampled only the prose front matter. Set either to a page
# count for a quick smoke run.
LOADTEST_FIXTURE_PAGES ?= 0
LOADTEST_SCAN_PAGES ?= 0
.PHONY: loadtest-fixtures
loadtest-fixtures:
	@test -n "$(SOURCE)" || { echo "SOURCE=<path to a real PDF> is required"; exit 1; }
	.venv/bin/python -m infra.loadtest.make_fixtures \
	  --source "$(SOURCE)" --pages $(LOADTEST_FIXTURE_PAGES) \
	  --scan-pages $(LOADTEST_SCAN_PAGES) --out $(K8S_LOADTEST)/fixtures
	@$(MAKE) --no-print-directory k8s-loadtest-fixtures-push

# Item 2.2's backlog: 20 real filings stratified across the measured service-time distribution,
# listed in infra/loadtest/backlog_sample.txt. Copies from the corpus into fixtures/backlog/, so
# a run selects it with LOADTEST_FIXTURE_DIR=/fixtures/backlog rather than naming 20 files.
.PHONY: loadtest-backlog
loadtest-backlog:
	.venv/bin/python -m infra.loadtest.make_backlog
	@$(MAKE) --no-print-directory k8s-loadtest-fixtures-push

# The fixtures PV's hostPath only names a real host directory if kind-cluster.yaml's extraMounts
# entry existed when the cluster was CREATED — extraMounts is a create-time property. On an
# older cluster the path is a directory inside the node container, so the PDFs have to be copied
# in, exactly as the results CSVs have to be copied out. Skipped when the bind mount is live:
# there source and destination are the same files, and `docker cp` would truncate each one while
# reading it.
K8S_NODE ?= copilot-control-plane
.PHONY: k8s-loadtest-fixtures-push
k8s-loadtest-fixtures-push:
	@find $(K8S_LOADTEST)/fixtures -name '*.pdf' | grep -q . || { \
	  echo "no fixtures to push; run: make loadtest-fixtures SOURCE=<a real filing>"; exit 1; }
	@docker inspect $(K8S_NODE) >/dev/null 2>&1 || { \
	  echo "node $(K8S_NODE) is not running; skipping push"; exit 0; }
	@if docker inspect $(K8S_NODE) --format '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' \
	   | grep -qx /mnt/loadtest-fixtures; then \
	  echo "$(K8S_NODE):/mnt/loadtest-fixtures is bind-mounted from the host — nothing to copy"; \
	else \
	  docker exec $(K8S_NODE) mkdir -p /mnt/loadtest-fixtures; \
	  : "trailing /. copies directory CONTENTS recursively, so backlog/ comes along"; \
	  docker cp $(K8S_LOADTEST)/fixtures/. $(K8S_NODE):/mnt/loadtest-fixtures/ || exit 1; \
	  echo "pushed to $(K8S_NODE):/mnt/loadtest-fixtures:"; \
	  docker exec $(K8S_NODE) find /mnt/loadtest-fixtures -name '*.pdf' | sed 's|^|  |'; \
	fi

# Deletes every loadtest-*@example.com user and everything that cascades from them, plus the
# Qdrant/OpenSearch/S3 entries for their documents (which do not cascade). Runs inside the api
# pod because it needs the cluster's own Qdrant/OpenSearch/S3 endpoints, and because the API's
# DELETE /v1/documents/{id} authenticates as the owning user — these are throwaway accounts
# with random passwords. Defaults to a dry run; pass YES=1 to actually delete.
.PHONY: k8s-loadtest-cleanup
k8s-loadtest-cleanup:
	$(eval POD := $(shell kubectl get pod -n copilot -l app.kubernetes.io/name=api \
	  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null))
	@test -n "$(POD)" || { echo "no api pod found"; exit 1; }
	# /tmp, not /app: the image ships infra/config/ only, and /app is root-owned while the
	# container runs as appuser. PYTHONPATH keeps `src` importable from the app root.
	kubectl cp infra/loadtest/cleanup.py copilot/$(POD):/tmp/loadtest_cleanup.py
	kubectl exec -n copilot $(POD) -- sh -c \
	  'cd /app && PYTHONPATH=/app:/tmp python -m loadtest_cleanup $(if $(YES),--yes,--dry-run)'

# Counterpart to the push: on such a cluster the CSVs land in the node container too, and a
# Completed pod cannot be `kubectl cp`-ed out of.
LOADTEST_RESULTS_DIR ?= $(K8S_LOADTEST)/results
.PHONY: k8s-loadtest-results
k8s-loadtest-results:
	@mkdir -p $(LOADTEST_RESULTS_DIR)
	docker cp $(K8S_NODE):/mnt/loadtest-results/. $(LOADTEST_RESULTS_DIR)/
	@ls -la $(LOADTEST_RESULTS_DIR)

# T13 (§8 item 1.4): the parse-timeout leak is a property of the wrap *firing*, not of when,
# so the run lowers the server's timeout instead of building a pathological PDF that takes ten
# minutes to parse. Turns a ~50 min run into a ~5 min one. `-restore` puts it back.
T13_PARSE_TIMEOUT ?= 60
.PHONY: k8s-loadtest-t13-timeout k8s-loadtest-t13-timeout-restore
k8s-loadtest-t13-timeout:
	kubectl set env deploy/worker-ingestion -n copilot \
	  DOCLING_PARSE_TIMEOUT_SECONDS=$(T13_PARSE_TIMEOUT)
	kubectl rollout status deploy/worker-ingestion -n copilot --timeout=5m
k8s-loadtest-t13-timeout-restore:
	kubectl set env deploy/worker-ingestion -n copilot DOCLING_PARSE_TIMEOUT_SECONDS-
	kubectl rollout status deploy/worker-ingestion -n copilot --timeout=5m

# §4.7's unresolved inconsistency, scoped to a run. Ingestion inherits the GLOBAL Celery limits
# (hard 450 / soft 360), which sit BELOW DOCLING_PARSE_TIMEOUT_SECONDS=600 — so the Celery hard
# limit kills a slow parse first, the least informative of the three failure paths: no
# parse_status, no partial result, and acks_late + INGEST_MAX_ATTEMPTS=2 then burns the slot a
# second time. Under the pre-optimization pipeline 3 of 50 real filings exceeded 450s.
#
# worker-ingestion is its own Deployment, so raising these here leaves worker-chat's hierarchy
# (960 > 450 > 360 > 180 > 60) untouched. Restores the ordering the audit asks for:
# visibility 960 > hard 900 > soft 850 > parse 600.
INGEST_HARD_LIMIT ?= 900
INGEST_SOFT_LIMIT ?= 850
.PHONY: k8s-loadtest-ingest-limits k8s-loadtest-ingest-limits-restore
k8s-loadtest-ingest-limits:
	kubectl set env deploy/worker-ingestion -n copilot \
	  CELERY_TASK_TIME_LIMIT_SECONDS=$(INGEST_HARD_LIMIT) \
	  CELERY_TASK_SOFT_TIME_LIMIT_SECONDS=$(INGEST_SOFT_LIMIT)
	kubectl rollout status deploy/worker-ingestion -n copilot --timeout=5m
k8s-loadtest-ingest-limits-restore:
	kubectl set env deploy/worker-ingestion -n copilot \
	  CELERY_TASK_TIME_LIMIT_SECONDS- CELERY_TASK_SOFT_TIME_LIMIT_SECONDS-
	kubectl rollout status deploy/worker-ingestion -n copilot --timeout=5m

.PHONY: k8s-sync-loadtest
k8s-sync-loadtest:
	cp infra/loadtest/locustfile.py infra/k8s/loadtest/locustfile.py
	cp infra/config/models.loadtest.yaml infra/k8s/loadtest/models.yaml

# Fails if the K8s loadtest copies have drifted. Run `make k8s-sync-loadtest` to fix.
.PHONY: k8s-check-loadtest
k8s-check-loadtest:
	@diff -q infra/loadtest/locustfile.py infra/k8s/loadtest/locustfile.py
	@diff -q infra/config/models.loadtest.yaml infra/k8s/loadtest/models.yaml

# Runs the Locust load test from inside the cluster (docs/notes/loadtest-concepts.md §7).
# Applying this overlay puts api + both workers into fake-LLM mode; `k8s-loadtest-clean`
# reverts. An initContainer refuses to generate load if that patch has not taken effect,
# so an accidental run cannot spend real money.
#
# Depends on k8s-deploy, not just k8s-set-images: MODELS_CONFIG_PATH is only honoured by
# config.py::load_models_config, so a cluster running an image built before that landed
# ignores the env var and silently serves the REAL models config. The gate catches it, but
# the fix is to ship current code — hence a full build+push here.
#
# Override per T-series scenario, e.g.:
#   make k8s-loadtest LOADTEST_USERS=100 LOADTEST_SPAWN_RATE=10 LOADTEST_DURATION=30m
# T2 (ramp, see infra/loadtest/locustfile.py::RampShape) is a shape, not a flat -u/-r/-t:
#   make k8s-loadtest LOADTEST_SHAPE=ramp LOADTEST_DURATION=30m
# LOADTEST_USERS/LOADTEST_SPAWN_RATE are ignored by locust once a shape is active — the ramp's
# own step knobs (LOADTEST_RAMP_*, see RampShape) take over instead.
# T3 (SSE hold) swaps the user class rather than the shape — pass LOADTEST_SHAPE= empty, since
# a shape would override the flat -u this scenario needs:
#   make k8s-loadtest LOADTEST_MODE=sse_hold LOADTEST_SHAPE= LOADTEST_USERS=40 \
#     LOADTEST_SPAWN_RATE=5 LOADTEST_DURATION=15m
K8S_LOADTEST := infra/k8s/loadtest
LOADTEST_USERS ?= 20
LOADTEST_SPAWN_RATE ?= 1
LOADTEST_DURATION ?= 10m
LOADTEST_TARGET_HOST ?= http://api:8000
LOADTEST_SHAPE ?=
# "ask" (default), "sse_hold" (T3) or "upload" (T11-T13) — picks the locustfile's user class,
# not a shape. "upload" needs fixtures on the host first: `make loadtest-fixtures SOURCE=...`.
LOADTEST_MODE ?= ask
LOADTEST_HOLD_FOR_S ?= 86400
# Ingestion track (docs/notes/loadtest-readiness-audit.md §8). One upload per user makes
# LOADTEST_USERS the backlog depth: -u 20 is item 2.2's 20-document queue.
LOADTEST_UPLOADS_PER_USER ?= 1
LOADTEST_INGEST_TIMEOUT_S ?= 1500
# Which fixture class(es) a run uploads. Empty draws from all of them, which is only right for
# a smoke test — see the note in loadtest-params.env.
#   make k8s-loadtest LOADTEST_MODE=upload LOADTEST_FIXTURES=normal.pdf ...
LOADTEST_FIXTURES ?=
# Which directory a run uploads from. /fixtures holds the three single-document classes;
# /fixtures/backlog holds item 2.2's 20-filing stratified sample (`make loadtest-backlog`).
LOADTEST_FIXTURE_DIR ?= /fixtures
# Think-time between one user's questions. This — not user count — is what actually drives
# utilisation: a user spends most of its cycle waiting, so halving these doubles arrival rate
# at the same user count (docs/notes/capacity-planning-concepts.md §1).
LOADTEST_MIN_WAIT ?= 180
LOADTEST_MAX_WAIT ?= 300
LOADTEST_RAMP_INITIAL_USERS ?= 1
LOADTEST_RAMP_STEP_USERS ?= 2
LOADTEST_RAMP_STEP_SECONDS ?= 30
LOADTEST_RAMP_MAX_USERS ?= 60
LOADTEST_RAMP_SPAWN_RATE ?= 10
# T4 (SpikeShape): spawn LOADTEST_SPIKE_USERS over _SPAWN_SECONDS, then hold for _HOLD.
# LOADTEST_DURATION/USERS/SPAWN_RATE are ignored while a shape is active.
LOADTEST_SPIKE_USERS ?= 50
LOADTEST_SPIKE_SPAWN_SECONDS ?= 10
LOADTEST_SPIKE_HOLD ?= 2m

.PHONY: k8s-loadtest
k8s-loadtest: k8s-sync-loadtest k8s-deploy
	@sed -i 's|^\( *newTag: \).*|\1$(GIT_SHA)|' $(K8S_LOADTEST)/kustomization.yaml
	@sed -i \
	  -e 's|^LOADTEST_TARGET_HOST=.*|LOADTEST_TARGET_HOST=$(LOADTEST_TARGET_HOST)|' \
	  -e 's|^LOADTEST_USERS=.*|LOADTEST_USERS=$(LOADTEST_USERS)|' \
	  -e 's|^LOADTEST_SPAWN_RATE=.*|LOADTEST_SPAWN_RATE=$(LOADTEST_SPAWN_RATE)|' \
	  -e 's|^LOADTEST_DURATION=.*|LOADTEST_DURATION=$(LOADTEST_DURATION)|' \
	  -e 's|^LOADTEST_SHAPE=.*|LOADTEST_SHAPE=$(LOADTEST_SHAPE)|' \
	  -e 's|^LOADTEST_MODE=.*|LOADTEST_MODE=$(LOADTEST_MODE)|' \
	  -e 's|^LOADTEST_HOLD_FOR_S=.*|LOADTEST_HOLD_FOR_S=$(LOADTEST_HOLD_FOR_S)|' \
	  -e 's|^LOADTEST_UPLOADS_PER_USER=.*|LOADTEST_UPLOADS_PER_USER=$(LOADTEST_UPLOADS_PER_USER)|' \
	  -e 's|^LOADTEST_INGEST_TIMEOUT_S=.*|LOADTEST_INGEST_TIMEOUT_S=$(LOADTEST_INGEST_TIMEOUT_S)|' \
	  -e 's|^LOADTEST_FIXTURES=.*|LOADTEST_FIXTURES=$(LOADTEST_FIXTURES)|' \
	  -e 's|^LOADTEST_FIXTURE_DIR=.*|LOADTEST_FIXTURE_DIR=$(LOADTEST_FIXTURE_DIR)|' \
	  -e 's|^LOADTEST_MIN_WAIT=.*|LOADTEST_MIN_WAIT=$(LOADTEST_MIN_WAIT)|' \
	  -e 's|^LOADTEST_MAX_WAIT=.*|LOADTEST_MAX_WAIT=$(LOADTEST_MAX_WAIT)|' \
	  -e 's|^LOADTEST_RAMP_INITIAL_USERS=.*|LOADTEST_RAMP_INITIAL_USERS=$(LOADTEST_RAMP_INITIAL_USERS)|' \
	  -e 's|^LOADTEST_RAMP_STEP_USERS=.*|LOADTEST_RAMP_STEP_USERS=$(LOADTEST_RAMP_STEP_USERS)|' \
	  -e 's|^LOADTEST_RAMP_STEP_SECONDS=.*|LOADTEST_RAMP_STEP_SECONDS=$(LOADTEST_RAMP_STEP_SECONDS)|' \
	  -e 's|^LOADTEST_RAMP_MAX_USERS=.*|LOADTEST_RAMP_MAX_USERS=$(LOADTEST_RAMP_MAX_USERS)|' \
	  -e 's|^LOADTEST_RAMP_SPAWN_RATE=.*|LOADTEST_RAMP_SPAWN_RATE=$(LOADTEST_RAMP_SPAWN_RATE)|' \
	  -e 's|^LOADTEST_SPIKE_USERS=.*|LOADTEST_SPIKE_USERS=$(LOADTEST_SPIKE_USERS)|' \
	  -e 's|^LOADTEST_SPIKE_SPAWN_SECONDS=.*|LOADTEST_SPIKE_SPAWN_SECONDS=$(LOADTEST_SPIKE_SPAWN_SECONDS)|' \
	  -e 's|^LOADTEST_SPIKE_HOLD=.*|LOADTEST_SPIKE_HOLD=$(LOADTEST_SPIKE_HOLD)|' \
	  $(K8S_LOADTEST)/loadtest-params.env
	kubectl delete job loadtest -n copilot --ignore-not-found
	# Two-phase apply, and the ordering is a money-safety property, not a nicety.
	#
	# Applying the Job in the SAME kubectl apply as the fake-LLM patches is a race that has
	# already cost real money once: the Job's pod starts immediately, its gate only checks
	# `api`, and api's new pod comes up long before worker-chat's six replicas finish rolling.
	# Locust then drives load into old workers still holding the REAL models config in their
	# cached router. Measured on 2026-09-10: $0.19 of live OpenAI spend over ~6 minutes,
	# with the gate reporting "fake LLM adapter confirmed active" the whole time.
	#
	# So: apply everything EXCEPT the Job, wait for all three deployments to finish rolling,
	# and only then create the Job. The rollout waits below are what actually enforce this —
	# they must stay BEFORE the Job is created, or the guarantee silently disappears again.
	kubectl kustomize $(K8S_LOADTEST) \
	  | python3 -c 'import sys,yaml; docs=[d for d in yaml.safe_load_all(sys.stdin) if d and not (d.get("kind")=="Job" and d["metadata"]["name"]=="loadtest")]; yaml.safe_dump_all(docs,sys.stdout)' \
	  | kubectl apply -f -
	kubectl rollout status deployment/api -n copilot --timeout=300s
	kubectl rollout status deployment/worker-chat -n copilot --timeout=600s
	kubectl rollout status deployment/worker-ingestion -n copilot --timeout=600s
	# Every LLM-calling workload is now serving the fake config; safe to generate load.
	kubectl kustomize $(K8S_LOADTEST) \
	  | python3 -c 'import sys,yaml; docs=[d for d in yaml.safe_load_all(sys.stdin) if d and d.get("kind")=="Job" and d["metadata"]["name"]=="loadtest"]; yaml.safe_dump_all(docs,sys.stdout)' \
	  | kubectl apply -f -
	@echo "streaming loadtest logs (Ctrl-C is safe, the Job keeps running)..."
	kubectl wait --for=create pod -l job-name=loadtest -n copilot --timeout=180s
	kubectl logs -f job/loadtest -n copilot

# Reverts the cluster out of fake-LLM mode and removes the Job. Run this when done, or the
# cluster keeps answering every question with the fake adapter.
.PHONY: k8s-loadtest-clean
k8s-loadtest-clean: k8s-deploy
	# k8s-deploy, not k8s-set-images: GIT_SHA includes a working-tree hash, so any edit since
	# the last build makes set-images point at a tag that was never pushed -> ImagePullBackOff
	# and a half-reverted cluster. Building is the only way to guarantee the tag resolves.
	#
	# loadtest is deleted here rather than by k8s-deploy because k8s-deploy doesn't know about
	# it; model-preload is already handled there.
	kubectl delete job loadtest -n copilot --ignore-not-found
	kubectl rollout status deployment/api -n copilot --timeout=300s
	kubectl rollout status deployment/worker-chat -n copilot --timeout=600s
	kubectl rollout status deployment/worker-ingestion -n copilot --timeout=600s

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
