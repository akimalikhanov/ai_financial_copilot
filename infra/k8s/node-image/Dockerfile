# Custom kind node image with the NVIDIA Container Toolkit baked in and registered as
# containerd's default runtime.
#
# Pin the base to the exact kindest/node image this cluster already runs (see
# infra/k8s/kind-cluster.yaml) so a kind version bump is a deliberate rebuild, not a silent
# toolkit/K8s version drift.
FROM kindest/node:v1.36.1@sha256:3489c7674813ba5d8b1a9977baea8a6e553784dab7b84759d1014dbd78f7ebd5

# Toolkit version pinned independently of the host's — the #1 GPU failure mode per
# docs/notes/gpu-in-kubernetes.md §5 is toolkit/driver skew between host and node.
ARG TOOLKIT_VERSION=1.18.1-1

# nvidia-container-toolkit's transitive deps (libnvidia-container-tools, libnvidia-container1)
# aren't auto-pinned by pinning the top-level package alone — the upstream repo has since moved
# past TOOLKIT_VERSION, so an unpinned `apt-get install nvidia-container-toolkit=X` fails with
# "Depends: libnvidia-container-tools (= X) but <newer> is to be installed". Pin the whole chain.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg2 ca-certificates \
 && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
      https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      nvidia-container-toolkit=${TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${TOOLKIT_VERSION} \
      libnvidia-container-tools=${TOOLKIT_VERSION} \
      libnvidia-container1=${TOOLKIT_VERSION} \
 && rm -rf /var/lib/apt/lists/* \
 && nvidia-ctk runtime configure --runtime=containerd \
      --config=/etc/containerd/config.toml --set-as-default

# Hardening pair (see `make k8s-gpu-preflight`): volume-mounts is the plugin-gated grant path
# (must be true); envvar-when-unprivileged is the self-grant bypass (must be false) — baking
# both into the image means a fresh cluster is hardened from first boot, not just after the
# host-side `nvidia-ctk config` step this repo also documents for the *host* daemon.
RUN nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true \
      --set accept-nvidia-visible-devices-envvar-when-unprivileged=false \
      --in-place --config-file=/etc/nvidia-container-runtime/config.toml

# Local registry: tells the node's containerd that
# pulls for localhost:5001/* should go to the kind-registry container over the kind network.
RUN mkdir -p /etc/containerd/certs.d/localhost:5001 \
 && printf '[host."http://kind-registry:5000"]\n' \
      > /etc/containerd/certs.d/localhost:5001/hosts.toml
