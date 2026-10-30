# CLAUDE.md

Guidance for coding agents (Claude Code and others) working in this repository.

## Project overview

AI Financial Copilot: a RAG chat application for financial documents. Users upload PDFs,
which are parsed, chunked and embedded, then ask questions answered with cited evidence from
those documents. Retrieval questions go through an agentic tool-calling loop.

## CRITICAL: docker compose invocation

Every `docker compose` command must include `--env-file ../../.env` (run from `infra/docker`).
This applies to every subcommand: up, down, build, logs, ps, exec, run.

- CORRECT: `docker compose --env-file ../../.env up -d`
- WRONG: `docker compose up -d`

## Development commands

Two ways to run the stack. Pick one.

**A) Host processes + dockerized infra** (hot reload):
```bash
# Infra + observability services first
cd infra/docker && docker compose --env-file ../../.env up -d

# One-time bootstraps
cd infra/docker && docker compose --env-file ../../.env run --rm garage-bootstrap
cd infra/docker && docker compose --env-file ../../.env run --rm langfuse-secrets

# API + chat worker + ingestion worker + UI in one terminal
make dev

# Or one at a time
make api                # FastAPI on :8000 (uvicorn --reload)
make worker             # Celery chat worker (queue: chat)
make worker-ingestion   # Celery ingestion worker (queue: ingestion)
make ui                 # Vite dev server (src/ui)
```

**B) Fully containerized stack** (production-like, no hot reload): the same compose file
defines `api`, `worker-chat`, `worker-ingestion` and `frontend`, built from
`infra/docker/Dockerfile.{api,worker,frontend}` with the repo root as context.
`docker compose --env-file ../../.env up -d` brings everything up. App containers reach infra
by in-network names (`pgbouncer:6432`, `redis-app:6379`, `qdrant:6333`) instead of the
`127.0.0.1` ports mode A uses.

Kubernetes (kind, and EKS later) runs from the Helm chart in `infra/k8s/`. See
[infra/k8s/README.md](infra/k8s/README.md).

```bash
# Checks: these make targets mirror CI, prefer them
make lint              # ruff check + ruff format --check + k8s copy drift checks
make typecheck         # pyright
make test-unit         # pytest tests/unit/
make test-integration  # pytest -m integration tests/integration/ (needs the Docker infra)
make test              # test-unit + test-integration
make test-cov          # test-unit with coverage (term + html report)
make helm-check        # helm lint + schema, kubeconform, helm-unittest

# One file or test
.venv/bin/python -m pytest tests/unit/test_citation_parser.py
.venv/bin/python -m pytest tests/unit/test_citation_parser.py::test_name

# Frontend
cd src/ui && npm run dev
cd src/ui && npm run build
cd src/ui && npx tsc --noEmit   # typecheck, as CI runs it
```

CI (`.github/workflows/ci.yml`) runs eight jobs in parallel: `lint`, `typecheck`,
`unit-tests`, `integration-tests`, `security` (gitleaks, pip-audit, semgrep), `docker`
(`docker compose config`, hadolint, Trivy config scan; no image builds), `helm`
(`make helm-check`) and `frontend-typecheck`. A `ci-success` job needs all eight and gates
the merge. Dependabot opens weekly grouped PRs for GitHub Actions and npm.

`.github/workflows/canary.yml` is manual-dispatch, not a PR gate. It runs
`src.eval.run_agent` (the same agent pipeline production serves) against
`src/eval/fixtures/canary_eval.json` on a pre-seeded eval environment with real LLM calls,
and fails if `src/eval/compare.py::compare()` finds a regression against
`src/eval/fixtures/runs/canary_baseline.json`. `--persist-db` / `--compare-to-latest-db`
store and compare runs in Postgres (`CanaryRun`, `CanaryRunResult`). It needs the secrets of
the `eval` GitHub environment.

## Python environment

- Python 3.12+, managed with `uv` (`uv.lock`), in a local `./.venv`.
- **Always run modules, never file paths, and never bare `python` or `pip`:**
  `.venv/bin/python -m src.some_package.some_module`
- Run from the repo root so module paths resolve.
- Pre-commit hooks: ruff (lint + format), pyright, gitleaks.
- Dev deps include `pytest-cov` and `respx` (mocks `httpx`, used for LLM/HTTP adapter tests).

## Architecture

The directory layout, with a line per package, is in
[README.md "Project structure"](README.md#project-structure). The agent loop is explained in
[docs/prod_docs/agent_design.md](docs/prod_docs/agent_design.md).

Two names that are easy to mix up: `src/services/router/` routes *queries* (retrieval or not,
which companies and documents), while `src/services/llm_router.py` routes *model IDs* to LLM
adapters using `infra/config/models.yaml`.

### Key data flows

**Chat request:** Frontend → POST `/v1/chat` → API creates the messages and enqueues a Celery
task → frontend opens SSE at `/v1/chat/stream?request_id=...` → chat worker
(`src/workers/chat_worker.py`, logic in `src/services/chat/tasks.py`) loads the request,
builds conversation context, scans user input for prompt injection, routes the query, and for
`retrieval` runs the agent loop (`src/services/chat/agent/`; the tool model writes each
search's semantic `query` and BM25 `keywords`) → renders the answering prompt → streams the
LLM response → persists and emits. Events go through a Redis stream; SSE delivers deltas,
citation spans, references, usage and a confidence badge.

**Document ingestion:** Frontend → POST `/v1/documents` (multipart) → API stores the PDF in
Garage (S3) and enqueues a Celery task → ingestion worker: Docling parse → enrich pictures →
chunk → summarize tables → embed → upsert Qdrant + index OpenSearch → update the Document
status.

**Query routing:** the LLM router (`src/services/router/`) returns a `route`
(`direct_answer`, `retrieval` or `out_of_scope`) and a `query_shape`, and scope resolution
turns company names into document IDs.

**Message feedback:** thumbs up/down in the UI → `feedback` router → `MessageFeedback` rows.

### Infrastructure (docker compose)

- **Core:** PostgreSQL (via pgBouncer), Redis (app + broker), Qdrant (vectors), OpenSearch
  (keyword) + OpenSearch Dashboards, TEI (embedding and reranking, GPU), Garage (S3, PDFs)
- **Observability:** Prometheus (:9090), Grafana (:3004), Fluent Bit (log shipping),
  Elasticsearch (:9201, logs), ClickHouse (Langfuse's trace store), Langfuse (:3003, web +
  worker)
- **App tier** (mode B only): `api`, `worker-chat`, `worker-ingestion`, `frontend`
- **One-time bootstraps:** `garage-bootstrap`, `langfuse-secrets`, `es-bootstrap`

## Code conventions

- The backend is async throughout (AsyncSession, async generators for streaming).
- Ruff: 100-char lines, double quotes, isort import order.
- Pydantic v2 for all schemas; SQLAlchemy 2.0 async ORM.
- Celery tasks use JSON serialization, `acks_late`, and queue routing (`chat` / `ingestion`).
- Citation markup in answers: `<claim refs="S1,S2">text</claim>`, parsed by
  `citation_parser.py`.
- LLM models are configured in `infra/config/models.yaml` (pricing, params, capabilities).
- Prompts are versioned `{name}_{version}.yaml` files in `prompts/`, loaded by
  `src/services/prompts/prompt_loader.py`. In use: `query_router_v5` (router,
  `QUERY_ROUTER_PROMPT_VERSION`), `system_v4_agent` or `system_v6_agent_analytical` (agent
  loop, picked by `ShapeConfig` from `query_shape`), `system_v5_agent_synthesis` (the
  answering prompt on every branch), `entity_disambiguator_v1`, `table_summarizer_v1`,
  `picture_enricher_*_v1`, `user_v1`. Older versions are kept alongside.
- Prometheus metrics from Celery prefork workers need `PROMETHEUS_MULTIPROC_DIR` set before
  `prometheus_client` is imported (see `make worker`, `make worker-ingestion`,
  `scripts/run_dev.sh`).
- **The application database is `app`, not `postgres`** (`APP_DB=app`, user `app`, via
  pgBouncer on `:6432`). The `postgres` database is the cluster default and has no app tables,
  so querying it returns zero rows instead of an error and a wrong `-d` looks like an empty
  system. For psql against the compose stack:
  `docker exec postgres psql -U postgres -d app`. Langfuse has its own database (`langfuse`).

## Key config files

- `.env` (copy of `.env.example`): environment variables
- `infra/config/models.yaml`: LLM model definitions; `models.loadtest.yaml` for load tests
- `infra/config/error_maps.yaml`: user-facing error messages
- `infra/config/vllm_config_llama.yaml`, `vllm_config_qwen.yaml`: vLLM server configs
- `prompts/*.yaml`: versioned prompt templates
- `infra/docker/docker-compose.yml`: infra, observability and app-tier services
- `infra/docker/prometheus/`, `infra/docker/grafana/`: metrics and dashboards
- `infra/docker/fluent-bit/fluent-bit.conf`, `infra/docker/elasticsearch/`: log shipping and
  index lifecycle
- `infra/docker/clickhouse/config.d/langfuse-cluster.xml`: Langfuse trace store
- `infra/k8s/chart/`, `infra/k8s/envs/`: Helm chart and per-environment values
  (`chart/files/app/app-config.env` holds app settings on Kubernetes)
