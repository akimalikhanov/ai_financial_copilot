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

# Ingest the sample corpus (idempotent). Needs `make worker-ingestion` running.
.PHONY: seed-sample
seed-sample:
	.venv/bin/python -m src.services.sample.seed

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
	{ git diff HEAD; \
	  git ls-files -o --exclude-standard -z | xargs -0 -I{} cat {} 2>/dev/null; \
	} | sha256sum | cut -c1-8)")
# Evaluated once, so every target in a run sees the same tag.
GIT_SHA := $(GIT_SHA)
REGISTRY := localhost:5001
# Newest tags kept per image by k8s-prune (tags a pod still references are always kept too).
K8S_KEEP_TAGS ?= 3
K8S_NODE := copilot-control-plane
# Every kubectl and helm call that reaches a cluster names it, so a deploy never lands on
# whatever context was current (`kind create cluster` switches it).
KUBE_CONTEXT ?= kind-copilot
KUBECTL = kubectl --context $(KUBE_CONTEXT)
HELM = helm --kube-context $(KUBE_CONTEXT)

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

# The build half of every deploy, so `k8s-loadtest` ships current code in the same tag.
.PHONY: k8s-build-push
k8s-build-push: docker-build-api docker-build-frontend docker-build-worker
	@# GIT_SHA passed down: an edit saved mid-build would otherwise change the tag under the push.
	$(MAKE) k8s-push-api k8s-push-frontend k8s-push-worker GIT_SHA=$(GIT_SHA)

# Full build+push+deploy loop (helm-deploy, below), then image pruning. Use the individual
# docker-build-*/k8s-push-* targets to iterate on one service. The four bootstrap Jobs are Helm
# hooks, re-run on every deploy; on a fresh cluster's first deploy model-preload downloads
# several GB, so give it time: `make k8s-deploy HELM_TIMEOUT=60m`.
.PHONY: k8s-deploy
k8s-deploy: helm-deploy
	$(MAKE) k8s-prune

# Drops old content-hashed image tags from the registry, the kind node and the host docker
# store, so repeated k8s-deploy runs don't accumulate ~17GB per tag in each. Dry run:
# `make k8s-prune PRUNE_ARGS=--dry-run`.
.PHONY: k8s-prune
k8s-prune:
	.venv/bin/python -m scripts.k8s_prune --registry $(REGISTRY) \
	  --repos copilot/api copilot/worker copilot/frontend --keep $(K8S_KEEP_TAGS) \
	  --node $(K8S_NODE) --registry-container kind-registry --kube-context $(KUBE_CONTEXT) \
	  $(PRUNE_ARGS)

# Helm chart (Stage 18), installed by k8s-deploy. infra/k8s/README.md is the operator guide.
CHART := infra/k8s/chart
ENV ?= kind
HELM_VALUES = -f infra/k8s/envs/values-$(ENV).yaml $(HELM_EXTRA_VALUES)
HELM_RENDER = helm template copilot $(CHART) -n copilot $(HELM_VALUES) \
  --set-string images.tag=$(GIT_SHA) --set-string secretsChecksum=$(SECRETS_CHECKSUM)

.PHONY: helm-render
helm-render:
	@$(HELM_RENDER)

# What a values file changes: kind vs eks, both rendered by Helm. MATCH narrows it to object
# names, e.g. MATCH='^postgres'.
.PHONY: helm-diff-envs
helm-diff-envs:
	@bash -c '.venv/bin/python -m scripts.k8s_render_diff --names kind eks \
	  <($(MAKE) -s helm-render ENV=kind) <($(MAKE) -s helm-render ENV=eks) \
	  $(if $(MATCH),--match "$(MATCH)")'

# The chart gate, also run by CI: lint + schema, kubeconform on each env's render, unit tests.
# Needs kubeconform and the helm-unittest plugin. Pin KUBECONFORM_K8S to the newest schema
# version not newer than the clusters (kind runs 1.36.1).
# Each entry is a comma-separated stack of envs/values-<name>.yaml files.
HELM_ENVS := kind eks kind,loadtest
KUBECONFORM_K8S := 1.36.1
.PHONY: helm-check
helm-check:
	helm lint --strict -n copilot $(CHART)
	@set -e; for e in $(HELM_ENVS); do \
	  echo "== $$e"; \
	  f=$$(for v in $$(echo $$e | tr , ' '); do printf -- '-f infra/k8s/envs/values-%s.yaml ' $$v; done); \
	  helm lint --strict -n copilot $(CHART) $$f --set-string images.tag=ci; \
	  out=$$(helm template copilot $(CHART) -n copilot $$f --set-string images.tag=ci); \
	  printf '%s\n' "$$out" | kubeconform -strict -summary -kubernetes-version $(KUBECONFORM_K8S); \
	done
	@echo "== loadtest chart"
	helm lint --strict -n copilot $(K8S_LOADTEST)/chart
	@set -e; out=$$(helm template loadtest $(K8S_LOADTEST)/chart -n copilot --set-string image.tag=ci); \
	  printf '%s\n' "$$out" | kubeconform -strict -summary -kubernetes-version $(KUBECONFORM_K8S)
	helm unittest $(CHART) $(K8S_LOADTEST)/chart

# Every deploy states all its values: never --reuse-values. --wait returns only once every
# Deployment has fully rolled out, which k8s-loadtest relies on.
HELM_TIMEOUT ?= 20m
HELM_DEPLOY = $(HELM) upgrade --install copilot $(CHART) -n copilot $(HELM_VALUES) \
  --set-string images.tag=$(GIT_SHA) --set-string secretsChecksum=$(SECRETS_CHECKSUM) \
  --wait --timeout $(HELM_TIMEOUT)

# Build, push, Secrets, then the release. HELM_EXTRA_VALUES layers more values files, e.g.
#   make helm-deploy KUBE_CONTEXT=kind-helm-smoke HELM_EXTRA_VALUES="-f infra/k8s/envs/values-smoke.yaml"
.PHONY: helm-deploy
helm-deploy: k8s-build-push k8s-secrets-apply
	$(HELM_DEPLOY)

.PHONY: print-secrets-checksum
print-secrets-checksum:
	@echo $(SECRETS_CHECKSUM)

# Runs the Locust load test from inside the cluster (docs/notes/loadtest-concepts.md §7):
# copilot in fake-LLM mode, then the `loadtest` release holding one run. `k8s-loadtest-clean`
# reverts. Knobs and scenario examples are with the LOADTEST_* defaults further down.
#
# Money safety is the order. copilot rolls out in fake-LLM mode under --wait, and only then is
# the Job's release created; its install refuses unless api, worker-chat and worker-ingestion
# all carry FAKE_LLM_ONLY=true, and the Job's own gates check again in the cluster.
#
# --wait (like `kubectl rollout status`) returns while old real-config pods are still
# Terminating, finishing what they hold for up to terminationGracePeriodSeconds (420s). That
# lag cost $0.0178 on 2026-09-20, so the drain loop waits until every api/worker pod still
# listed, Terminating ones included, carries FAKE_LLM_ONLY=true.
LOADTEST_DRAIN_TIMEOUT_S ?= 600
LOADTEST_PODS := app.kubernetes.io/name in (api,worker-chat,worker-ingestion)
LOADTEST_PARAMS := TARGET_HOST USERS SPAWN_RATE DURATION CHAT_MODEL CHAT_EMAIL CHAT_PASSWORD \
  SSE_TIMEOUT_S MIN_WAIT MAX_WAIT MODE FIXTURE_DIR FIXTURES UPLOADS_PER_USER UPLOAD_MIN_WAIT \
  UPLOAD_MAX_WAIT INGEST_TIMEOUT_S HOLD_FOR_S SHAPE RAMP_INITIAL_USERS RAMP_STEP_USERS \
  RAMP_STEP_SECONDS RAMP_MAX_USERS RAMP_SPAWN_RATE SPIKE_USERS SPIKE_SPAWN_SECONDS SPIKE_HOLD
# Only the ones set on the command line or in the environment, as shell-quoted NAME=value
# words; the loadtest chart's values.yaml holds the defaults. Sent as JSON, so commas
# (LOADTEST_FIXTURES) and quotes survive.
shell-quote = '$(subst ','\'',$(1))'
LOADTEST_SET = $(foreach p,$(LOADTEST_PARAMS),$(if $(filter command environment,$(origin LOADTEST_$(p))),$(call shell-quote,LOADTEST_$(p)=$(LOADTEST_$(p)))))
LOADTEST_PARAMS_JSON = .venv/bin/python -c \
  'import json,sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:])))' $(LOADTEST_SET)

.PHONY: k8s-loadtest
k8s-loadtest: k8s-sync-loadtest k8s-build-push k8s-secrets-apply
	$(HELM_DEPLOY) -f infra/k8s/envs/values-loadtest.yaml \
	  --set-string loadtest.faults.fakeLlmLatencyMs=$(LOADTEST_FAKE_LLM_LATENCY_MS) \
	  --set-string loadtest.faults.opensearchHost=$(LOADTEST_OPENSEARCH_HOST)
	@echo "waiting for real-config pods to drain..."
	@end=$$(( $$(date +%s) + $(LOADTEST_DRAIN_TIMEOUT_S) )); \
	while :; do \
	  out=$$($(KUBECTL) get pod -n copilot -l '$(LOADTEST_PODS)' -o jsonpath='{range .items[*]}{.metadata.name}={range .spec.containers[0].env[?(@.name=="FAKE_LLM_ONLY")]}{.value}{end}{"\n"}{end}') \
	    && ! printf '%s\n' "$$out" | grep -v -e '=true$$' -e '^$$' | grep -q . && break; \
	  [ $$(date +%s) -lt $$end ] || { echo "real-config pods still present after $(LOADTEST_DRAIN_TIMEOUT_S)s"; exit 1; }; \
	  sleep 5; done
	$(HELM) uninstall loadtest -n copilot --ignore-not-found --wait
	params=$$($(LOADTEST_PARAMS_JSON)) && $(HELM) install loadtest $(K8S_LOADTEST)/chart -n copilot \
	    --set-string image.tag=$(GIT_SHA) --set-json "params=$$params"
	@echo "streaming loadtest logs (Ctrl-C is safe, the Job keeps running)..."
	$(KUBECTL) wait --for=create pod -l job-name=loadtest -n copilot --timeout=180s
	$(KUBECTL) logs -f job/loadtest -n copilot

# Back out of fake-LLM mode. The Job goes first, and its pod must be gone before the real config
# returns, or Locust's last requests would reach real-config workers. --reset-values: the
# result never depends on what the load-test release held. Helm prunes toxiproxy.
.PHONY: k8s-loadtest-clean
k8s-loadtest-clean: k8s-build-push k8s-secrets-apply
	$(HELM) uninstall loadtest -n copilot --ignore-not-found --wait
	@if $(KUBECTL) get pod -l job-name=loadtest -n copilot -o name | grep -q .; then \
	  $(KUBECTL) wait --for=delete pod -l job-name=loadtest -n copilot --timeout=120s; fi
	$(HELM_DEPLOY) --reset-values

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

# Cluster-level add-on, not part of the app (the copilot release never touches it): installs
# once per cluster lifetime and survives every future k8s-deploy. Only needed
# again after kind delete cluster. kind-cluster.yaml already has the extraPortMappings
# (80/443) and ingress-ready=true node label this manifest's controller expects.
.PHONY: k8s-ingress-nginx
k8s-ingress-nginx:
	$(KUBECTL) apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
	$(KUBECTL) wait --namespace ingress-nginx --for=condition=ready pod \
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

# NVIDIA's device plugin chart as its own release in kube-system, never a dependency of the
# copilot chart (whose namespace guard pins everything to copilot). The chart version also
# pins the plugin image tag. --repo fetches the chart without a local `helm repo add`.
NVDP_CHART_REPO ?= https://nvidia.github.io/k8s-device-plugin
NVDP_CHART_VERSION ?= 0.19.3
NVDP_VALUES ?= infra/k8s/addons/nvidia-device-plugin-kind.yaml

.PHONY: k8s-gpu-plugin
k8s-gpu-plugin:
	$(HELM) upgrade --install nvdp nvidia-device-plugin --repo $(NVDP_CHART_REPO) \
	  --version $(NVDP_CHART_VERSION) -n kube-system -f $(NVDP_VALUES) --wait --timeout 5m
	@echo "waiting for nvidia.com/gpu to appear on the node..."
	@# After a plugin swap the kubelet reports 0 until the new one registers, so 0 doesn't count.
	@for i in $$(seq 1 60); do \
	  gpus="$$($(KUBECTL) get node copilot-control-plane -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null)"; \
	  if [ -n "$$gpus" ] && [ "$$gpus" != "0" ]; then \
	    echo "GPU allocatable: $$($(KUBECTL) get node copilot-control-plane -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"; exit 0; fi; \
	  sleep 2; done; \
	  echo "TIMEOUT: nvidia.com/gpu never became allocatable. kubectl -n kube-system logs -l app.kubernetes.io/instance=nvdp --all-containers"; exit 1

.PHONY: k8s-down
k8s-down:
	kind delete cluster --name copilot

# Helm reads only files inside a chart, so the charts carry copies of files the compose stack
# also uses. The k8s-sync-* targets refresh the copies from the source of truth; the matching
# k8s-check-* targets fail on drift and run in `lint`. Everything else under $(CHART)/files/
# (app-config.env, prometheus.yml, the K8s-only dashboards, ...) is the original.

# Source of truth: infra/scripts/. Run after editing any of them.
.PHONY: k8s-sync-initdb
k8s-sync-initdb:
	mkdir -p $(CHART)/files/data/initdb
	cp infra/scripts/db_init/00_db_init.sh $(CHART)/files/data/initdb/00_db_init.sh
	cp infra/scripts/db_init/01_create_app_tables.sh $(CHART)/files/data/initdb/01_create_app_tables.sh
	cp infra/scripts/pgbouncer-entrypoint.sh $(CHART)/files/data/pgbouncer-entrypoint.sh

.PHONY: k8s-check-initdb
k8s-check-initdb:
	@diff -q infra/scripts/db_init/00_db_init.sh $(CHART)/files/data/initdb/00_db_init.sh
	@diff -q infra/scripts/db_init/01_create_app_tables.sh $(CHART)/files/data/initdb/01_create_app_tables.sh
	@diff -q infra/scripts/pgbouncer-entrypoint.sh $(CHART)/files/data/pgbouncer-entrypoint.sh

# Source of truth: infra/docker/elasticsearch/.
.PHONY: k8s-sync-es-bootstrap
k8s-sync-es-bootstrap:
	mkdir -p $(CHART)/files/search/es-bootstrap
	cp infra/docker/elasticsearch/bootstrap.sh $(CHART)/files/search/es-bootstrap/bootstrap.sh
	cp infra/docker/elasticsearch/ilm-policy.json $(CHART)/files/search/es-bootstrap/ilm-policy.json
	cp infra/docker/elasticsearch/index-template.json $(CHART)/files/search/es-bootstrap/index-template.json

.PHONY: k8s-check-es-bootstrap
k8s-check-es-bootstrap:
	@diff -q infra/docker/elasticsearch/bootstrap.sh $(CHART)/files/search/es-bootstrap/bootstrap.sh
	@diff -q infra/docker/elasticsearch/ilm-policy.json $(CHART)/files/search/es-bootstrap/ilm-policy.json
	@diff -q infra/docker/elasticsearch/index-template.json $(CHART)/files/search/es-bootstrap/index-template.json

# Source of truth: infra/docker/grafana/dashboards/. Not synced: resources.json exists only in
# the chart, and the chart's worker-health.json adds GPU panels (dcgm-exporter) compose lacks.
K8S_SHARED_DASHBOARDS := api-overview agentic-rag cost-tokens logs-explorer eval-canary
.PHONY: k8s-sync-dashboards
k8s-sync-dashboards:
	@for d in $(K8S_SHARED_DASHBOARDS); do \
	  cp infra/docker/grafana/dashboards/$$d.json $(CHART)/files/observability/grafana/dashboards/$$d.json; \
	done

.PHONY: k8s-check-dashboards
k8s-check-dashboards:
	@for d in $(K8S_SHARED_DASHBOARDS); do \
	  diff -q infra/docker/grafana/dashboards/$$d.json $(CHART)/files/observability/grafana/dashboards/$$d.json || exit 1; \
	done

# The model-preload hook warms the hf-model-cache PVC with Docling's models, so the first PDF
# upload doesn't stall on a multi-GB download inside the Celery task. It runs on every deploy
# and is a no-op once cached; this is the deploy with room for a fresh cluster's first download.
.PHONY: k8s-preload-models
k8s-preload-models:
	$(MAKE) k8s-deploy HELM_TIMEOUT=60m

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
	$(eval POD := $(shell $(KUBECTL) get pod -n copilot -l app.kubernetes.io/name=api \
	  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null))
	@test -n "$(POD)" || { echo "no api pod found"; exit 1; }
	# /tmp, not /app: the image ships infra/config/ only, and /app is root-owned while the
	# container runs as appuser. PYTHONPATH keeps `src` importable from the app root.
	$(KUBECTL) cp infra/loadtest/cleanup.py copilot/$(POD):/tmp/loadtest_cleanup.py
	$(KUBECTL) exec -n copilot $(POD) -- sh -c \
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
	$(KUBECTL) set env deploy/worker-ingestion -n copilot \
	  DOCLING_PARSE_TIMEOUT_SECONDS=$(T13_PARSE_TIMEOUT)
	$(KUBECTL) rollout status deploy/worker-ingestion -n copilot --timeout=5m
k8s-loadtest-t13-timeout-restore:
	$(KUBECTL) set env deploy/worker-ingestion -n copilot DOCLING_PARSE_TIMEOUT_SECONDS-
	$(KUBECTL) rollout status deploy/worker-ingestion -n copilot --timeout=5m

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
	$(KUBECTL) set env deploy/worker-ingestion -n copilot \
	  CELERY_TASK_TIME_LIMIT_SECONDS=$(INGEST_HARD_LIMIT) \
	  CELERY_TASK_SOFT_TIME_LIMIT_SECONDS=$(INGEST_SOFT_LIMIT)
	$(KUBECTL) rollout status deploy/worker-ingestion -n copilot --timeout=5m
k8s-loadtest-ingest-limits-restore:
	$(KUBECTL) set env deploy/worker-ingestion -n copilot \
	  CELERY_TASK_TIME_LIMIT_SECONDS- CELERY_TASK_SOFT_TIME_LIMIT_SECONDS-
	$(KUBECTL) rollout status deploy/worker-ingestion -n copilot --timeout=5m

# T8's proxy image. The node pulls only from the kind registry (its containerd hosts.toml),
# so ghcr.io has to be mirrored through the host's docker. Once per cluster; the registry is
# a container with --restart=always, so it survives everything short of `kind delete`.
TOXIPROXY_IMAGE := ghcr.io/shopify/toxiproxy:2.12.0
.PHONY: k8s-loadtest-toxiproxy-image
k8s-loadtest-toxiproxy-image: k8s-registry
	docker pull $(TOXIPROXY_IMAGE)
	docker tag $(TOXIPROXY_IMAGE) $(REGISTRY)/toxiproxy:2.12.0
	docker push $(REGISTRY)/toxiproxy:2.12.0

# T9 (dependency down). Scaling the StatefulSet, not deleting the pod: a deleted pod is
# recreated within seconds and the outage window is too short to ask a question into, which
# is the whole measurement. Scale-to-0 is also a graceful terminate, so the PVCs come back
# intact — these carry the real corpus, not throwaway data.
#
# Each target names one backend so the three phases of T9 (qdrant alone, opensearch alone,
# both) are composable from the command line rather than encoded as a fourth target.
.PHONY: k8s-loadtest-t9-kill-qdrant k8s-loadtest-t9-kill-opensearch k8s-loadtest-t9-restore
k8s-loadtest-t9-kill-qdrant:
	$(KUBECTL) scale statefulset/qdrant -n copilot --replicas=0
	$(KUBECTL) wait --for=delete pod/qdrant-0 -n copilot --timeout=180s
k8s-loadtest-t9-kill-opensearch:
	$(KUBECTL) scale statefulset/opensearch -n copilot --replicas=0
	$(KUBECTL) wait --for=delete pod/opensearch-0 -n copilot --timeout=300s
k8s-loadtest-t9-restore:
	$(KUBECTL) scale statefulset/qdrant -n copilot --replicas=1
	$(KUBECTL) scale statefulset/opensearch -n copilot --replicas=1
	$(KUBECTL) rollout status statefulset/qdrant -n copilot --timeout=300s
	$(KUBECTL) rollout status statefulset/opensearch -n copilot --timeout=600s

# Chart copies of infra/loadtest/locustfile.py and infra/config/models.loadtest.yaml (see the
# k8s-sync-* note above). k8s-loadtest syncs them before every run.
.PHONY: k8s-sync-loadtest
k8s-sync-loadtest:
	mkdir -p $(CHART)/files/loadtest $(K8S_LOADTEST)/chart/files
	cp infra/config/models.loadtest.yaml $(CHART)/files/loadtest/models.yaml
	cp infra/loadtest/locustfile.py $(K8S_LOADTEST)/chart/files/locustfile.py

.PHONY: k8s-check-loadtest
k8s-check-loadtest:
	@diff -q infra/config/models.loadtest.yaml $(CHART)/files/loadtest/models.yaml
	@diff -q infra/loadtest/locustfile.py $(K8S_LOADTEST)/chart/files/locustfile.py

# Knobs for `make k8s-loadtest` (the target sits with the other Helm targets above). The
# defaults here mirror infra/k8s/loadtest/chart/values.yaml, and only the ones set on the
# command line or in the environment are sent. Override per T-series scenario, e.g.:
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
# Which fixture class(es) a run uploads, comma-separated. Empty draws from all of them, which
# is only right for a smoke test: the classes differ ~4x in service time, so a mix measures
# nothing in particular.
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
# Log in as a seeded account rather than registering a throwaway one. Required for any run
# whose measurement depends on retrieval actually returning chunks (T5, T8, T9, T12) — an
# account owning no documents matches nothing in either backend, so a "degraded" answer and
# a healthy one are indistinguishable. Empty keeps the throwaway behaviour.
LOADTEST_CHAT_EMAIL ?=
LOADTEST_CHAT_PASSWORD ?=
# Client-side SSE budget. Raise it above the scenario's worst legitimate service time, or the
# client times out on answers the server is still producing and the run measures this number.
# T7 needs it above 300s, its pass bar.
LOADTEST_SSE_TIMEOUT_S ?= 180

# --- T7-T9 fault injection (docs/notes/loadtest-readiness-audit.md §5) ----------------------
# Both default to "no fault", so every other scenario is unaffected by their presence.
#
# T7 (slow LLM): a fixed FakeAdapter latency instead of its sampled profile, which is the
# knob the concepts note (§6) prefers over standing up toxiproxy for the LLM leg.
#   make k8s-loadtest LOADTEST_MODE=ask LOADTEST_SHAPE= LOADTEST_FAKE_LLM_LATENCY_MS=20000 \
#     LOADTEST_USERS=20 LOADTEST_SPAWN_RATE=2 LOADTEST_DURATION=20m
LOADTEST_FAKE_LLM_LATENCY_MS ?=
# T8 (slow OpenSearch): "toxiproxy" routes api + worker-chat through the chart's
# templates/loadtest/toxiproxy.yaml, which holds an 8s downstream latency toxic. OpenSearch
# itself stays real, as concepts §6 requires. Run `make k8s-loadtest-toxiproxy-image` once per cluster first.
LOADTEST_OPENSEARCH_HOST ?= opensearch

# Gitignored env files, one per Secret; k8s-secrets-apply generates any that are missing.
K8S_ENV_DIR := infra/k8s/secrets/kind
K8S_SECRET_NAMES := postgres redis garage app grafana clickhouse langfuse
# Changes whenever any env file changes; the chart puts it on every secret-reading pod.
SECRETS_CHECKSUM = $(shell cat $(foreach s,$(K8S_SECRET_NAMES),$(K8S_ENV_DIR)/$(s).env) \
  2>/dev/null | sha256sum | cut -c1-16)

# Creates or updates the Secrets the chart references by name (`<name>-secret`), from the env
# files. Idempotent. Server-side apply, so the secret data isn't duplicated into a
# last-applied-configuration annotation. Every k8s-deploy runs it first.
.PHONY: k8s-secrets-apply
k8s-secrets-apply: k8s-secrets
	@# Create-only: the namespace is outside the release, and applying it would rewrite its labels.
	$(KUBECTL) get namespace copilot >/dev/null 2>&1 || $(KUBECTL) create namespace copilot
	@for s in $(K8S_SECRET_NAMES); do \
	  kubectl create secret generic $$s-secret -n copilot \
	    --from-env-file=$(K8S_ENV_DIR)/$$s.env --dry-run=client -o yaml \
	    | $(KUBECTL) apply --server-side --field-manager=make -f - ; \
	done

.PHONY: k8s-secrets
k8s-secrets:
	@mkdir -p $(K8S_ENV_DIR)
	@test -f $(K8S_ENV_DIR)/postgres.env || { \
	  { echo "POSTGRES_USER=postgres"; \
	    echo "POSTGRES_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	    echo "APP_DB_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	    echo "LANGFUSE_DB_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	  } > $(K8S_ENV_DIR)/postgres.env; chmod 600 $(K8S_ENV_DIR)/postgres.env; \
	  echo "generated $(K8S_ENV_DIR)/postgres.env"; }
	@test -f $(K8S_ENV_DIR)/redis.env || { \
	  { echo "REDIS_PASSWORD=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	  } > $(K8S_ENV_DIR)/redis.env; chmod 600 $(K8S_ENV_DIR)/redis.env; \
	  echo "generated $(K8S_ENV_DIR)/redis.env"; }
	@test -f $(K8S_ENV_DIR)/garage.env || { \
	  { echo "garage_rpc_secret=$$(openssl rand -hex 32)"; \
	    echo "garage_admin_token=$$(openssl rand -base64 32)"; \
	    echo "garage_metrics_token=$$(openssl rand -base64 32)"; \
	    echo "garage_s3_access_key_id=GK$$(openssl rand -hex 12)"; \
	    echo "garage_s3_secret_access_key=$$(openssl rand -hex 32)"; \
	  } > $(K8S_ENV_DIR)/garage.env; chmod 600 $(K8S_ENV_DIR)/garage.env; \
	  echo "generated $(K8S_ENV_DIR)/garage.env"; }
	@# The S3 key pair above is pre-generated and imported by the garage-bootstrap Job (stage-17
	@# plan Phase 9, concept 4) rather than created at runtime, so it's already known here — write
	@# it straight into app.env instead of a post-bootstrap sync step. app.env gains more keys
	@# (JWT_SECRET, LLM provider keys, ...) once the app tier lands; this only owns the AWS_* pair.
	@test -f $(K8S_ENV_DIR)/app.env || touch $(K8S_ENV_DIR)/app.env && chmod 600 $(K8S_ENV_DIR)/app.env
	@grep -q '^AWS_ACCESS_KEY_ID=' $(K8S_ENV_DIR)/app.env || { \
	  ACCESS_KEY="$$(grep '^garage_s3_access_key_id=' $(K8S_ENV_DIR)/garage.env | cut -d= -f2)"; \
	  SECRET_KEY="$$(grep '^garage_s3_secret_access_key=' $(K8S_ENV_DIR)/garage.env | cut -d= -f2)"; \
	  { echo "AWS_ACCESS_KEY_ID=$$ACCESS_KEY"; \
	    echo "AWS_SECRET_ACCESS_KEY=$$SECRET_KEY"; \
	  } >> $(K8S_ENV_DIR)/app.env; \
	  echo "wrote AWS_* creds into $(K8S_ENV_DIR)/app.env"; }
	@# Phase 10 (app tier): JWT_SECRET is generated fresh; APP_DB_PASSWORD/REDIS_PASSWORD are
	@# copied from the postgres/redis secret files generated above (single source of truth,
	@# api container just needs its own secretRef with the same values). LLM provider keys and
	@# the Langfuse keypair are external/not-yet-generated — left blank for manual entry.
	@grep -q '^JWT_SECRET=' $(K8S_ENV_DIR)/app.env || { \
	  APP_DB_PASSWORD="$$(grep '^APP_DB_PASSWORD=' $(K8S_ENV_DIR)/postgres.env | cut -d= -f2)"; \
	  REDIS_PASSWORD="$$(grep '^REDIS_PASSWORD=' $(K8S_ENV_DIR)/redis.env | cut -d= -f2)"; \
	  { echo "JWT_SECRET=$$(openssl rand -base64 32 | tr -d '/+=')"; \
	    echo "OPENAI_API_KEY="; \
	    echo "GOOGLE_API_KEY="; \
	    echo "HF_TOKEN="; \
	    echo "APP_DB_USER=app"; \
	    echo "APP_DB_PASSWORD=$$APP_DB_PASSWORD"; \
	    echo "REDIS_PASSWORD=$$REDIS_PASSWORD"; \
	    echo "LANGFUSE_PUBLIC_KEY="; \
	    echo "LANGFUSE_SECRET_KEY="; \
	  } >> $(K8S_ENV_DIR)/app.env; \
	  echo "wrote JWT_SECRET/APP_DB_*/REDIS_PASSWORD into $(K8S_ENV_DIR)/app.env (fill in LLM/Langfuse keys manually)"; }
	@test -f $(K8S_ENV_DIR)/grafana.env || { \
	  { echo "GRAFANA_USER=admin"; \
	    echo "GRAFANA_PASS=$$(openssl rand -base64 24 | tr -d '/+=')"; \
	  } > $(K8S_ENV_DIR)/grafana.env; chmod 600 $(K8S_ENV_DIR)/grafana.env; \
	  echo "generated $(K8S_ENV_DIR)/grafana.env"; }
	@# Phase 16: ClickHouse credentials. Read by the ClickHouse StatefulSet *and* by both
	@# Langfuse pods from this same Secret, so the two can never drift apart.
	@test -f $(K8S_ENV_DIR)/clickhouse.env || { \
	  { echo "CLICKHOUSE_USER=clickhouse"; \
	    echo "CLICKHOUSE_PASSWORD=$$(openssl rand -base64 32 | tr -d '/+=')"; \
	  } > $(K8S_ENV_DIR)/clickhouse.env; chmod 600 $(K8S_ENV_DIR)/clickhouse.env; \
	  echo "generated $(K8S_ENV_DIR)/clickhouse.env"; }
	@# Phase 16: Langfuse. Mirrors infra/scripts/langfuse_bootstrap.sh. Connection URLs are
	@# composed here (rather than assembled in the pod) because Langfuse wants single DSN
	@# strings and K8s env can't interpolate one Secret key into another.
	@test -f $(K8S_ENV_DIR)/langfuse.env || { \
	  LANGFUSE_DB_PASSWORD="$$(grep '^LANGFUSE_DB_PASSWORD=' $(K8S_ENV_DIR)/postgres.env | cut -d= -f2)"; \
	  REDIS_PASSWORD="$$(grep '^REDIS_PASSWORD=' $(K8S_ENV_DIR)/redis.env | cut -d= -f2)"; \
	  CH_USER="$$(grep '^CLICKHOUSE_USER=' $(K8S_ENV_DIR)/clickhouse.env | cut -d= -f2)"; \
	  CH_PASS="$$(grep '^CLICKHOUSE_PASSWORD=' $(K8S_ENV_DIR)/clickhouse.env | cut -d= -f2)"; \
	  S3_KEY="$$(grep '^garage_s3_access_key_id=' $(K8S_ENV_DIR)/garage.env | cut -d= -f2)"; \
	  S3_SECRET="$$(grep '^garage_s3_secret_access_key=' $(K8S_ENV_DIR)/garage.env | cut -d= -f2)"; \
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
	  } > $(K8S_ENV_DIR)/langfuse.env; chmod 600 $(K8S_ENV_DIR)/langfuse.env; \
	  echo "generated $(K8S_ENV_DIR)/langfuse.env"; }
	@# The app tier authenticates to Langfuse with the seeded project's key pair, so the two
	@# blank placeholders written into app.env in Phase 10 are filled from langfuse.env here.
	@grep -q '^LANGFUSE_PUBLIC_KEY=.' $(K8S_ENV_DIR)/app.env || { \
	  PK="$$(grep '^LANGFUSE_INIT_PROJECT_PUBLIC_KEY=' $(K8S_ENV_DIR)/langfuse.env | cut -d= -f2)"; \
	  SK="$$(grep '^LANGFUSE_INIT_PROJECT_SECRET_KEY=' $(K8S_ENV_DIR)/langfuse.env | cut -d= -f2)"; \
	  sed -i "s|^LANGFUSE_PUBLIC_KEY=.*|LANGFUSE_PUBLIC_KEY=$$PK|; s|^LANGFUSE_SECRET_KEY=.*|LANGFUSE_SECRET_KEY=$$SK|" \
	    $(K8S_ENV_DIR)/app.env; \
	  echo "wrote LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY into $(K8S_ENV_DIR)/app.env"; }
