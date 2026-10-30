# copilot chart

Helm chart for the AI Financial Copilot stack. Environment settings live in `../envs/`. How to
deploy, roll back, rotate a secret or run a load test is in [`../README.md`](../README.md); this
file is about how the chart is built.

```bash
make helm-render                 # helm template with envs/values-kind.yaml (ENV=eks for EKS)
make helm-diff-envs              # kind vs EKS: should show only the seams listed below
make helm-check                  # the CI gate: lint + schema, kubeconform, helm unittest
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
`envs/values-smoke.yaml` layers on kind for install rehearsals on a throwaway cluster with no
GPU: everything that needs the GPU, or waits on something that does, is scaled to zero.

Still out of `values-eks.yaml`, for Stage 19 to decide: `langfuseWeb.nextauthUrl`, OpenSearch's
`DISABLE_SECURITY_PLUGIN`, and the RDS / S3 / IRSA seams.

## Files

Helm reads only files inside the chart, so `files/` holds copies of files the compose stack
also uses: the initdb and PgBouncer scripts (`infra/scripts/`), es-bootstrap
(`infra/docker/elasticsearch/`), five Grafana dashboards (`infra/docker/grafana/dashboards/`)
and the load-test models file. `make k8s-sync-*` refreshes them, and `make lint` fails on drift.
Everything else under `files/` is the original, `files/app/app-config.env` included: new app
settings go there.

## How the chart behaves

- Every object carries `app.kubernetes.io/name` (from `copilot.labels`). Selectors and pod
  template labels kept their Stage 17 values, because they are immutable on a live object.
- ConfigMaps and Secrets have fixed names. Pods roll on a change through `checksum/config` and
  `checksum/secrets` annotations instead. The chart never renders a Secret: it references
  `<name>-secret`, which `make k8s-secrets-apply` creates.
- `es-bootstrap`, `garage-bootstrap`, `model-preload` and `seed-sample` are
  post-install/post-upgrade hooks, re-run on every deploy and idempotent. `seed-sample` has
  weight 1, so it runs after the other three; ingestion needs the bucket and the models.
- Grafana has no `wait-for-es-bootstrap` gate. Under `--wait` a hook runs only after the
  release is ready, so the gate would deadlock a fresh install. Logs panels error for the ~15s
  until es-bootstrap creates the `copilot-logs` alias.
- `hf-model-cache`, `tei-data`, `tei-embedder-data`, `prometheus-data` and `grafana-data` carry
  `helm.sh/resource-policy: keep`, so `helm uninstall` leaves them in place. The StatefulSets'
  PVCs come from `volumeClaimTemplates`, which Helm never deletes either.
- The ClusterRoles and ClusterRoleBindings are named `copilot-<name>`, because cluster-scoped
  names are global.

## Load tests

Two releases with different lifetimes:

- `copilot` + `envs/values-loadtest.yaml` puts api, worker-chat and worker-ingestion in fake-LLM
  mode (one switch, `loadtest.fakeLlm.enabled`, read by the `copilot.fakeLlm.*` helpers all three
  include), deploys toxiproxy, and skips the seed-sample hook. `loadtest.faults.*` sets the T7/T8
  faults and refuses to render outside fake-LLM mode. A fault's env entry renders only when set.
- `loadtest` (`infra/k8s/loadtest/chart`) is one run: the Job, its RBAC and ConfigMaps, and the
  results and fixtures PV/PVC pairs, which are kept on uninstall so the Retain PVs rebind.

`make k8s-loadtest` deploys copilot with `--wait`, waits until no real-config api/worker pod is
left (Terminating ones included), then reinstalls `loadtest`. The install refuses (a `lookup`)
unless all three Deployments carry `FAKE_LLM_ONLY=true`, and the Job's own gates check again in
the cluster. Only the `LOADTEST_*` variables you set are passed; the defaults live in the
loadtest chart's `values.yaml`. `make k8s-loadtest-clean` uninstalls the run first, then
redeploys with `--reset-values`, which also prunes toxiproxy. The three pods roll on a
models-file edit via `checksum/loadtest-models`.

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

## Cutover

On 2026-10-29 Helm adopted the live Kustomize-built stack with `--take-ownership
--force-conflicts`, after a fresh-install rehearsal on a throwaway cluster with
`values-smoke.yaml`. Every PVC kept its UID, and every PV is `Retain` since. The Kustomize tree
was deleted the same day; `git log -- infra/k8s/base` has it. Lessons for the next adoption
(Stage 19):

- **Drop the old field manager after adopting.** Where Helm applied the value that was already
  live, the field stayed co-owned with `kubectl-client-side-apply`, so the first upgrade that
  changed it (the image tag) failed with a conflict. Removing that manager's `managedFields`
  entries, and its `last-applied-configuration` annotation, from the 62 adopted objects made
  Helm the sole owner. That is a metadata patch; no pod restarted.
- **Check what the live state equals with the server, as Helm.** `helm get manifest copilot -n
  copilot | kubectl diff --server-side --field-manager=helm --force-conflicts -f -` shows only
  Helm's own `meta.helm.sh/*` annotations, which the manifest leaves out. A plain `kubectl diff`
  compares against the old `last-applied-configuration` instead.
- **`--wait` also waits for every PVC to bind.** With a `WaitForFirstConsumer` storage class
  (kind's `standard`, EKS `gp3`), a standalone PVC binds only once a pod uses it. On the smoke
  cluster `tei-data`, `tei-embedder-data` and `hf-model-cache` have no pod, so the install
  times out unless a throwaway pod mounts them first. A real install has their pods.
- **Adoption never deletes a field the chart leaves out.** Grafana's gate and OpenSearch
  Dashboards' old curl image were removed through Kustomize first, so the live objects already
  matched the chart.
- **The rollback drill works.** `helm rollback copilot 1 -n copilot --wait` returned the images
  to revision 1's tag without re-running the hooks.

Left over on kind-copilot from Kustomize, to delete by hand: the hashed `*-<10 chars>`
ConfigMaps and Secrets with no `app.kubernetes.io/managed-by` label; the `prometheus`,
`fluent-bit` and `kube-state-metrics` ClusterRoles and ClusterRoleBindings (replaced by
`copilot-*`); and the kubectl-created load-test objects (ServiceAccount `loadtest`, Role and
RoleBinding `loadtest-rollout-reader`, the `loadtest-*` ConfigMaps, PVCs and PVs), which `helm
install loadtest` can't adopt. The load-test hostPath data on the node stays, because the PVs
are `Retain`. Leave `Namespace/copilot` alone: Helm doesn't manage it, so no Helm command can
delete it.
