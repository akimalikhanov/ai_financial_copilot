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

{{/*
Privileged initContainer setting vm.max_map_count on the node, for Lucene's mmaps. Callers wrap
it in `if .Values.sysctlInit.enabled`. Privileged is required: runc masks /proc/sys read-only for
non-privileged containers even with SYS_ADMIN.
*/}}
{{- define "copilot.sysctlInitContainer" -}}
- name: sysctl
  image: {{ .Values.sysctlInit.image }}
  command: ["sh", "-c", "sysctl -w vm.max_map_count=262144"]
  securityContext:
    privileged: true
{{- end }}

{{/* Pod-level GPU scheduling. The container adds nvidia.com/gpu to its own limits. */}}
{{- define "copilot.gpuScheduling" -}}
{{- if .Values.gpu.enabled -}}
{{- with .Values.gpu.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.gpu.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}
{{- end }}

{{/*
ConfigMap data from a KEY=VALUE chart file, plus overrides. Argument:
(dict "root" $ "path" "files/..." "overrides" map). Inside, $ is that dict.
*/}}
{{- define "copilot.envFileData" -}}
{{- $data := dict -}}
{{- range $line := splitList "\n" (.root.Files.Get .path) -}}
{{- $l := trim $line -}}
{{- if and $l (not (hasPrefix "#" $l)) -}}
{{- $kv := regexSplit "=" $l 2 -}}
{{- if ne (len $kv) 2 -}}{{- fail (printf "%s: expected KEY=VALUE, got %q" $.path $l) -}}{{- end -}}
{{- $_ := set $data (index $kv 0) (index $kv 1) -}}
{{- end -}}
{{- end -}}
{{- range $k, $v := (.overrides | default dict) -}}
{{- $_ := set $data $k (toString $v) -}}
{{- end -}}
{{- range $k, $v := $data }}
{{ $k }}: {{ $v | quote }}
{{- end -}}
{{- end }}

{{/* initContainer that blocks until pg_isready. Argument: (dict "root" $ "host" "pgbouncer" "port" 6432 "userVar" "APP_DB_USER"). */}}
{{- define "copilot.waitForPostgres" -}}
- name: wait-for-{{ .host }}
  image: {{ .root.Values.waitImages.postgres }}
  command:
    - sh
    - -c
    - |
      until pg_isready -h {{ .host }} -p {{ .port }} -U "${{ .userVar }}"; do
        echo "waiting for {{ .host }}:{{ .port }}"; sleep 2
      done
  envFrom:
    - secretRef: { name: {{ .root.Values.secrets.postgres }} }
  resources:
    requests: { cpu: "10m", memory: "16Mi" }
{{- end }}

{{/* initContainer that blocks until a Redis instance answers PING. Argument: (dict "root" $ "host" "redis-app"). */}}
{{- define "copilot.waitForRedis" -}}
- name: wait-for-{{ .host }}
  image: {{ .root.Values.waitImages.redis }}
  command:
    - sh
    - -c
    - |
      until redis-cli -h {{ .host }} -p 6379 -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; do
        echo "waiting for {{ .host }}:6379"; sleep 2
      done
  envFrom:
    - secretRef: { name: {{ .root.Values.secrets.redis }} }
  resources:
    requests: { cpu: "10m", memory: "16Mi" }
{{- end }}

{{/*
initContainer that blocks until an HTTP endpoint answers. Argument:
(dict "root" $ "host" "tei" "port" 80 "path" "/health" ["anyStatus" true]).
anyStatus accepts any HTTP response, e.g. Garage's 403 to anonymous requests.
*/}}
{{- define "copilot.waitForHttp" -}}
- name: wait-for-{{ .host }}
  image: {{ .root.Values.waitImages.curl }}
  command:
    - sh
    - -c
    - |
      {{- if .anyStatus }}
      until curl -sS -o /dev/null http://{{ .host }}:{{ .port }}{{ .path }}; do
      {{- else }}
      until curl -fsS http://{{ .host }}:{{ .port }}{{ .path }} >/dev/null 2>&1; do
      {{- end }}
        echo "waiting for {{ .host }}:{{ .port }}"; sleep 2
      done
  resources:
    requests: { cpu: "10m", memory: "16Mi" }
{{- end }}

{{/*
initContainer that blocks until each host:port accepts TCP connections, in order. No client or
credentials needed. Argument: (dict "root" $ "name" "deps" "targets" (list "postgres:5432" ...)).
*/}}
{{- define "copilot.waitForTcp" -}}
- name: wait-for-{{ .name }}
  image: {{ .root.Values.waitImages.busybox }}
  command:
    - sh
    - -c
    - |
      for t in {{ join " " .targets }}; do
        until nc -z "${t%:*}" "${t#*:}"; do echo "waiting for $t"; sleep 2; done
      done
  resources:
    requests: { cpu: "10m", memory: "16Mi" }
{{- end }}

{{/* storageClassName, omitted when persistence.storageClass is "" (cluster default). */}}
{{- define "copilot.storageClass" -}}
{{- with .Values.persistence.storageClass -}}
storageClassName: {{ . | quote }}
{{- end -}}
{{- end }}
