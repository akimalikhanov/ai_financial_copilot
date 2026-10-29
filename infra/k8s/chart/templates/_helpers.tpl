{{/* Namespace for every namespaced object. Config files in files/ hardcode "copilot". */}}
{{- define "copilot.namespace" -}}
{{- if ne .Release.Namespace "copilot" -}}
{{- fail "the copilot chart installs only into namespace copilot" -}}
{{- end -}}
{{- .Release.Namespace -}}
{{- end }}

{{/* Object metadata labels. Argument: (dict "root" $ "name" "<app name>"). */}}
{{- define "copilot.labels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/part-of: copilot
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/managed-by: {{ .root.Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .root.Chart.Name .root.Chart.Version }}
{{- end }}

{{/* Selector labels. Immutable on live objects: never add a key. */}}
{{- define "copilot.selectorLabels" -}}
app.kubernetes.io/name: {{ .name }}
{{- end }}

{{/* Pod template labels. No chart version here, or every chart bump restarts every pod. */}}
{{- define "copilot.podLabels" -}}
{{ include "copilot.selectorLabels" . }}
app.kubernetes.io/part-of: copilot
{{- with .component }}
app.kubernetes.io/component: {{ . }}
{{- end }}
{{- end }}

{{/* One of our own images: copilot/api, copilot/worker, copilot/frontend. */}}
{{- define "copilot.appImage" -}}
{{- $i := .root.Values.images -}}
{{- if $i.registry }}{{ $i.registry }}/{{ end }}copilot/{{ .name }}:{{ $i.tag | toString }}
{{- end }}

{{/* storageClassName, omitted when persistence.storageClass is "" (cluster default). */}}
{{- define "copilot.storageClass" -}}
{{- with .Values.persistence.storageClass -}}
storageClassName: {{ . | quote }}
{{- end -}}
{{- end }}
