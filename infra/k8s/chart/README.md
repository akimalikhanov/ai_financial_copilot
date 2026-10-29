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
