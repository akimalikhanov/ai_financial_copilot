# Kubernetes

The stack runs on a local kind cluster (`kind-copilot`) as one Helm release, `copilot`, in the
`copilot` namespace. EKS (Stage 19) renders from the same chart with a different values file.

```
infra/k8s/
├── chart/            the copilot chart (how it's built: chart/README.md)
├── envs/             values-kind, values-eks, values-loadtest, values-smoke
├── loadtest/chart/   one load-test run, its own release
├── secrets/kind/     gitignored env files, one per Secret (*.env.example are committed)
├── addons/           values for cluster add-ons installed from upstream charts (NVIDIA device plugin)
├── kind-cluster.yaml the cluster: GPU node image, ingress ports, load-test mounts
└── node-image/       Dockerfile for that node image
```

Every make target that reaches a cluster passes `--kube-context $(KUBE_CONTEXT)` (default
`kind-copilot`), never the current context.

## From a clean checkout

Once per machine:

1. The host GPU setup. `make k8s-gpu-preflight` checks it and prints the fix for each failure.
2. `make k8s-node-image`, the kind node image with the NVIDIA toolkit and the registry config.
3. `uv sync`, for the `.venv` that `make k8s-deploy`'s image prune runs in.
4. docker, kind, kubectl and helm 4 on the PATH.

Then:

```bash
make k8s-up                       # cluster, local registry, GPU device plugin, ingress-nginx
make k8s-secrets                  # generates the env files in secrets/kind/
# fill in OPENAI_API_KEY (and GOOGLE_API_KEY, HF_TOKEN if used) in secrets/kind/app.env
make k8s-deploy HELM_TIMEOUT=60m  # build, push, Secrets, helm upgrade --install --wait, prune
```

The first deploy takes long because the `model-preload` hook downloads several GB of Docling
models; later deploys find them cached. The app is then on <http://localhost>.

## Environments

| | `values-kind.yaml` | `values-eks.yaml` (render-only) |
|---|---|---|
| images | `localhost:5001` | ECR (placeholder account) |
| storage | kind `standard`, small PVCs | `gp3`, full sizes |
| GPU | time-sliced, no taint | taint toleration |
| ingress | `nginx` | `alb` + annotations |
| sysctl initContainer | on | off (node bootstrap sets it) |
| dcgm-exporter, opensearch-dashboards | on | off |

`make helm-diff-envs` shows the full difference; anything outside these rows is a bug.
`values-loadtest.yaml` and `values-smoke.yaml` layer on top of kind (see below and
[chart/README.md](chart/README.md)).

## Secrets

No secret is in git or in a values file. Each Secret's data lives in a gitignored env file in
`secrets/kind/` (mode 600). `make k8s-secrets` generates any missing file, and
`make k8s-secrets-apply` creates the `<name>-secret` Secrets from them with server-side apply.
`make k8s-deploy` runs both. The chart only references the Secrets by name, and stamps a
checksum of all the env files on every pod that reads one, so a changed file rolls those pods.

To rotate a secret: edit the env file, then `make k8s-deploy`. Two catches:

- Some values are copies. `app.env` and `langfuse.env` repeat the database, Redis, ClickHouse
  and S3 credentials from the other files, and `app.env` repeats Langfuse's project keys, so
  change every copy.
- A data store keeps what it was created with. Postgres reads `POSTGRES_PASSWORD` and the role
  passwords only on an empty volume, so change the password in the database first
  (`ALTER ROLE ... PASSWORD ...`), then in the file. `garage-bootstrap` skips a key name that
  already exists, so a new S3 key pair needs the old key deleted in Garage first.

## Deploy and roll back

```bash
make k8s-deploy                                    # current working tree
helm --kube-context kind-copilot history copilot -n copilot
helm --kube-context kind-copilot rollback copilot <REVISION> -n copilot --wait
helm --kube-context kind-copilot get values copilot -n copilot --revision <REVISION>
```

The image tag is the commit plus a hash of uncommitted changes, so every saved edit deploys as
a new image. A rollback restores the revision's manifests but doesn't re-run the hooks, and it
works only while that revision's images still exist: `make k8s-prune` (part of `k8s-deploy`)
keeps the newest `K8S_KEEP_TAGS` tags (default 3) plus any a pod still uses. For a second
cluster, use `make helm-deploy KUBE_CONTEXT=...`, which skips the prune.

## Load tests

```bash
make k8s-loadtest LOADTEST_USERS=50 LOADTEST_DURATION=20m   # fake-LLM stack + one run
make k8s-loadtest-results                                    # copy the CSVs out of the node
make k8s-loadtest-clean                                      # back to the real stack
```

`k8s-loadtest` switches api and both workers to the fake LLM and waits until no real-config
pod is left before starting the run, so a load test never spends money. The stack stays in
fake-LLM mode until `make k8s-loadtest-clean`. The `LOADTEST_*` knobs are documented next to
their defaults in the Makefile. The T9/T13/ingest-limit targets change chart objects on
purpose (`kubectl scale`, `kubectl set env`); run their `-restore` target before the next
deploy.

## CI

The `helm` job runs `make helm-check` (lint, schema, kubeconform, helm-unittest) on every PR
and feeds `ci-success`. As of 2026-10-29, `ci-success` is red on master for `docker`,
`unit-tests` and `security`, and master has no branch protection, so a red check blocks
nothing yet. `make lint` also fails when a chart's copy of a compose file drifts.

## Never

- **`helm uninstall copilot`** without reading what it deletes: every Deployment, StatefulSet,
  Service, ConfigMap and RBAC object, so the whole stack goes down. The data survives: the five
  `keep` PVCs and the StatefulSets' PVCs stay, and so do the Secrets, which the release doesn't
  own.
- **`kubectl delete namespace copilot`.** It deletes every PVC and Secret. The PVs are `Retain`,
  so the data stays on the node, but getting it back means rebinding each PV by hand.
- **`--reuse-values`.** It carries the last release's values forward, a load test's fake-LLM
  switch included, and ignores new defaults in the chart's `values.yaml`. The make targets pass
  the values files on every deploy.
- **`kubectl apply` or `kubectl edit` on an object the chart manages.** The edit takes
  ownership of the fields it touches: the next deploy fails on a conflict, or the edit lingers
  unseen in a field the chart doesn't set. Change the chart or a values file instead.
