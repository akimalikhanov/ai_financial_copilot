{{/* Namespace: the Job dials api:8000 and checks the copilot release's Deployments. */}}
{{- define "loadtest.namespace" -}}
{{- if ne .Release.Namespace "copilot" -}}
{{- fail "the loadtest chart installs only into namespace copilot" -}}
{{- end -}}
{{- .Release.Namespace -}}
{{- end }}

{{/* Object metadata labels. Argument: the root context. */}}
{{- define "loadtest.labels" -}}
app.kubernetes.io/name: loadtest
app.kubernetes.io/part-of: copilot
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end }}

{{/* Kept on helm uninstall: a deleted PVC leaves its Retain PV Released, and it never rebinds. */}}
{{- define "loadtest.keep" -}}
helm.sh/resource-policy: keep
{{- end }}

{{/*
Install-time check: every LLM-calling Deployment must already be in fake-LLM mode. Reads the
spec, so it relies on `helm upgrade --wait` for copilot having finished first. Offline
(`helm template`, CI) lookup returns nothing and the check is skipped; the Job's own gates
still run in the cluster.
*/}}
{{- define "loadtest.requireFakeLlm" -}}
{{- if lookup "v1" "Namespace" "" .Release.Namespace -}}
{{- range $name := list "api" "worker-chat" "worker-ingestion" -}}
{{- $d := lookup "apps/v1" "Deployment" $.Release.Namespace $name -}}
{{- if not $d -}}
{{- fail (printf "Deployment %s not found: deploy copilot with envs/values-loadtest.yaml first" $name) -}}
{{- end -}}
{{- $ok := false -}}
{{- range $c := $d.spec.template.spec.containers -}}
{{- range $e := ($c.env | default list) -}}
{{- if and (eq $e.name "FAKE_LLM_ONLY") (eq (toString $e.value) "true") -}}{{- $ok = true -}}{{- end -}}
{{- end -}}
{{- end -}}
{{- if not $ok -}}
{{- fail (printf "%s is not in fake-LLM mode: deploy copilot with envs/values-loadtest.yaml first" $name) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}
