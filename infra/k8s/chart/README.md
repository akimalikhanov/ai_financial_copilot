# copilot chart

Helm chart for the AI Financial Copilot stack. Environment settings live in `../envs/`.

```bash
make helm-render                 # helm template with envs/values-kind.yaml
make helm-diff-kustomize         # Kustomize vs Helm, object by object (MATCH='^postgres' narrows it)
```

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
- `hf-model-cache`, `tei-data` and `tei-embedder-data` carry `helm.sh/resource-policy: keep`
  (scrubbed by the diff script), so `helm uninstall` leaves them in place.
- Grafana loses its `wait-for-es-bootstrap` initContainer. As a hook under `--wait`,
  es-bootstrap runs only after Grafana is ready, so the gate would deadlock a fresh install.
  Grafana now starts before the `copilot-logs` alias exists, and logs panels error for the
  ~15s until es-bootstrap finishes.
