# copilot chart

Helm chart for the AI Financial Copilot stack. Environment settings live in `../envs/`.

```bash
make helm-render                 # helm template with envs/values-kind.yaml (ENV=eks for EKS)
make helm-diff-kustomize         # Kustomize vs Helm, object by object (MATCH='^postgres' narrows it)
make helm-diff-envs              # kind vs EKS: should show only the seams listed below
make helm-check                  # the CI gate: lint + schema, kubeconform, helm unittest
make helm-loadtest               # load test on the Helm stack (live after the cutover)
```

`helm-check` needs [kubeconform](https://github.com/yannh/kubeconform) and the
[helm-unittest](https://github.com/helm-unittest/helm-unittest) plugin; CI pins both in the
`helm` job of `.github/workflows/ci.yml`.

## Environments

`envs/values-kind.yaml` runs today. `envs/values-eks.yaml` is render-only until Stage 19; its
ECR account and region are placeholders. The two differ only in these seams:

- image registry
- storage class (`gp3` on EKS) and PVC sizes (kind shrinks them to match its live claims)
- GPU taint toleration (EKS)
- ingress class and annotations (`nginx` / `alb`)
- the sysctl initContainer (off on EKS, where the node bootstrap sets `vm.max_map_count`)
- `dcgm-exporter` and `opensearch-dashboards` (kind only)

Anything else in `make helm-diff-envs` means environment logic leaked into a template.

`envs/values-loadtest.yaml` layers on kind for a load test only (see below).

Still out of `values-eks.yaml`, for Stage 19 to decide: `langfuseWeb.nextauthUrl`, OpenSearch's
`DISABLE_SECURITY_PLUGIN`, and the RDS / S3 / IRSA seams.

## Load tests

Two releases with different lifetimes:

- `copilot` + `envs/values-loadtest.yaml` puts api, worker-chat and worker-ingestion in fake-LLM
  mode (one switch, `loadtest.fakeLlm.enabled`, read by the `copilot.fakeLlm.*` helpers all three
  include), deploys toxiproxy, and skips the seed-sample hook. `loadtest.faults.*` sets the T7/T8
  faults and refuses to render outside fake-LLM mode.
- `loadtest` (`infra/k8s/loadtest/chart`) is one run: the Job, its RBAC and ConfigMaps, and the
  results and fixtures PV/PVC pairs, which are kept on uninstall so the Retain PVs rebind.

`make helm-loadtest` deploys copilot with `--wait`, then reinstalls `loadtest`. The install
refuses (a `lookup`) unless all three Deployments carry `FAKE_LLM_ONLY=true`, and the Job's own
gates check again in the cluster. It takes the same `LOADTEST_*` variables as `k8s-loadtest`;
only the ones you set are passed, and the defaults live in the loadtest chart's `values.yaml`.
`make helm-loadtest-clean` uninstalls the run first, then redeploys with `--reset-values`, which
also prunes toxiproxy. Neither target is live until the cutover; until then
`k8s-loadtest` / `k8s-loadtest-clean` drive Kustomize.

Against `kubectl kustomize infra/k8s/loadtest`, the expected differences on top of those under
"Intended changes vs Kustomize":

- The fault env entries render only when set; Kustomize always set `FAKE_LLM_LATENCY_MS=""`,
  which the app reads the same as unset. Env order differs, since Kustomize prepends patch entries.
- `loadtest-*` ConfigMaps lose their hash suffix. The three pods roll on a models-config edit via
  `checksum/loadtest-models`; the Job is new every run, so it needs none.
- `loadtest-params` defaults are the Makefile's `LOADTEST_*` defaults, not whatever the last
  `k8s-loadtest` run wrote into `loadtest-params.env`.
- No `seed-sample` hook under load-test values.
- Comment lines inside the Job's and toxiproxy's scripts are shorter; the commands are identical.

At the cutover, the live load-test objects (Job, ServiceAccount, Role, RoleBinding, ConfigMaps,
the two PVCs and PVs, toxiproxy) were created by kubectl, so `helm install loadtest` can't adopt
them. Delete them first; the hostPath data on the node stays, because the PVs are `Retain`.

## Checks

- `values.schema.json` rejects unknown keys at the top level and inside the seam objects
  (`images`, `secrets`, `persistence`, `gpu`, `ingress`, `sysctlInit`, `appConfig`) and the four
  `enabled`-flag components, so `gpu.enable: true` fails lint and install rather than doing
  nothing. Other components are checked only as objects. `images.tag` must be a string, which is
  why every make target passes it with `--set-string`.
- `helm-check` covers kind, eks and kind + loadtest, plus the loadtest chart.
- `tests/` holds the helm-unittest suites: the GPU, storage, ingress, image and load-test seams, the keep
  policy, the namespace guard, schema typos, and the frozen fields (selectors, volumeClaimTemplate
  names) that would break `helm upgrade` if changed. A suite renders only the templates it lists,
  so a template whose `checksum/config` includes a ConfigMap template needs that one listed too.
- kubeconform validates against the Kubernetes 1.36.1 schemas (`KUBECONFORM_K8S` in the Makefile).
- The `helm` CI job feeds `ci-success`. As of 2026-10-29, `ci-success` is red on master for
  `docker`, `unit-tests` and `security`, and master has no branch protection, so a red check
  blocks nothing yet.
- trivy's config scan skips `infra/k8s`, which covers this chart.

## Intended changes vs Kustomize

Every difference `make helm-diff-kustomize` reports that is not a template bug goes here. Each one
is applied to the live stack before the cutover.

- Object metadata gains `app.kubernetes.io/name` (from `copilot.labels`). Selectors and pod
  template labels are unchanged.
- ConfigMaps and Secrets lose their hash suffix. Pods roll on change via `checksum/config` and
  `checksum/secrets` instead. The Secrets are created by `make k8s-secrets-apply`.
- `es-bootstrap`, `garage-bootstrap`, `model-preload` and `seed-sample` are
  post-install/post-upgrade hooks, re-run on every deploy. `seed-sample` has weight 1, so it runs
  after the other three; ingestion needs the bucket and the models. Their pods also gain
  `app.kubernetes.io/name`. Delete the live Jobs before the cutover; a hook can't adopt them.
- `wait-for-*` initContainers come from shared helpers. The Garage gate's script loses its
  comment lines, so `api` and `worker-ingestion` see a pod template change.
- OpenSearch Dashboards' gate uses the shared helper too: curl 8.14.1 instead of 8.11.1, and no
  `grep '"status"'` (`curl -f` already fails on a non-2xx).
- `hf-model-cache`, `tei-data`, `tei-embedder-data`, `prometheus-data` and `grafana-data` carry
  `helm.sh/resource-policy: keep` (scrubbed by the diff script), so `helm uninstall` leaves them
  in place.
- The ClusterRoles and ClusterRoleBindings for `prometheus`, `fluent-bit` and
  `kube-state-metrics` are renamed `copilot-<name>`, because cluster-scoped names are global.
  The cutover creates the new pairs; delete the old three pairs after it.
- `dcgm-exporter` and `opensearch-dashboards` render only when `dcgmExporter.enabled` /
  `opensearchDashboards.enabled` are true (kind only).
- Grafana loses its `wait-for-es-bootstrap` initContainer. As a hook under `--wait`,
  es-bootstrap runs only after Grafana is ready, so the gate would deadlock a fresh install.
  Grafana now starts before the `copilot-logs` alias exists, and logs panels error for the
  ~15s until es-bootstrap finishes.
