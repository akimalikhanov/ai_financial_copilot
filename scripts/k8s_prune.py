"""Prune old app image tags from the local registry, the kind node and the host docker store.

Every `make k8s-deploy` pushes a new content-hashed tag per image and nothing removes the old
ones. Keeps, per repo: every tag a pod currently references, plus the newest --keep tags by
image creation time. A registry digest is only deleted when no kept tag points at it.

    .venv/bin/python -m scripts.k8s_prune --registry localhost:5001 \
        --repos copilot/api copilot/worker copilot/frontend --keep 3 \
        --node copilot-control-plane --registry-container kind-registry \
        --kube-context kind-copilot [--dry-run]
"""

import argparse
import json
import subprocess
import urllib.error
import urllib.request

MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}


def _get(url: str, accept: str | None = None) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(req) as resp:
        return resp.read(), dict(resp.headers)


def _run(cmd: list[str], check: bool = True) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=check).stdout


def pods_image_refs(context: str) -> set[str]:
    pods = json.loads(_run(["kubectl", "--context", context, "get", "pods", "-A", "-o", "json"]))[
        "items"
    ]
    refs: set[str] = set()
    for pod in pods:
        for key in ("containers", "initContainers"):
            refs.update(c["image"] for c in pod["spec"].get(key, []))
    return refs


def tag_digest_and_created(base: str, repo: str, tag: str) -> tuple[str, str]:
    """Return (manifest digest the tag points at, image creation timestamp)."""
    body, headers = _get(f"{base}/v2/{repo}/manifests/{tag}", MANIFEST_ACCEPT)
    digest = headers["Docker-Content-Digest"]
    manifest = json.loads(body)
    if manifest.get("mediaType") in INDEX_TYPES:
        # buildkit attaches attestation manifests (platform unknown/unknown); skip those.
        child = next(
            m for m in manifest["manifests"] if m.get("platform", {}).get("os") != "unknown"
        )
        body, _ = _get(f"{base}/v2/{repo}/manifests/{child['digest']}", MANIFEST_ACCEPT)
        manifest = json.loads(body)
    config, _ = _get(f"{base}/v2/{repo}/blobs/{manifest['config']['digest']}")
    return digest, json.loads(config)["created"]


def prune_registry(args: argparse.Namespace, in_use: set[str]) -> dict[str, set[str]]:
    base = f"http://{args.registry}"
    kept: dict[str, set[str]] = {}
    deleted_any = False
    for repo in args.repos:
        tags = json.loads(_get(f"{base}/v2/{repo}/tags/list")[0]).get("tags") or []
        info = {t: tag_digest_and_created(base, repo, t) for t in tags}
        newest = sorted(tags, key=lambda t: info[t][1], reverse=True)[: args.keep]
        keep = {t for t in tags if f"{args.registry}/{repo}:{t}" in in_use} | set(newest)
        keep_digests = {info[t][0] for t in keep}
        doomed = {info[t][0] for t in tags if t not in keep} - keep_digests
        print(f"registry {repo}: {len(tags)} tags, keeping {len(keep)}, deleting {len(doomed)}")
        kept[repo] = keep
        if args.dry_run:
            continue
        for digest in sorted(doomed):
            req = urllib.request.Request(f"{base}/v2/{repo}/manifests/{digest}", method="DELETE")
            try:
                urllib.request.urlopen(req).close()
                deleted_any = True
            except urllib.error.HTTPError as exc:
                print(f"  delete {digest[:19]} failed: {exc.code}")
    if deleted_any:
        print("registry garbage-collect ...")
        out = _run(
            [
                "docker",
                "exec",
                args.registry_container,
                "registry",
                "garbage-collect",
                "--delete-untagged",
                "/etc/distribution/config.yml",
            ],
            check=False,
        )
        print(out.strip().splitlines()[-1] if out.strip() else "  (no output)")
    return kept


def prune_node(args: argparse.Namespace, kept: dict[str, set[str]], in_use: set[str]) -> None:
    images = json.loads(_run(["docker", "exec", args.node, "crictl", "images", "-o", "json"]))
    keep_refs = {f"{args.registry}/{r}:{t}" for r, tags in kept.items() for t in tags} | in_use
    prefixes = tuple(f"{args.registry}/{r}:" for r in args.repos)
    doomed = [
        tag
        for img in images["images"]
        for tag in img.get("repoTags", [])
        if tag.startswith(prefixes) and tag not in keep_refs
    ]
    print(f"kind node: removing {len(doomed)} image tags")
    if args.dry_run or not doomed:
        return
    _run(["docker", "exec", args.node, "crictl", "rmi", *doomed], check=False)


def prune_host(args: argparse.Namespace, kept: dict[str, set[str]], in_use: set[str]) -> None:
    listed = _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]).split()
    keep_refs = {f"{args.registry}/{r}:{t}" for r, tags in kept.items() for t in tags} | in_use
    prefixes = tuple(f"{args.registry}/{r}:" for r in args.repos)
    doomed = [t for t in listed if t.startswith(prefixes) and t not in keep_refs]
    print(f"host docker: removing {len(doomed)} image tags")
    if args.dry_run or not doomed:
        return
    _run(["docker", "rmi", *doomed], check=False)
    _run(["docker", "image", "prune", "-f"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--registry", required=True, help="host:port of the local registry")
    parser.add_argument("--repos", nargs="+", required=True)
    parser.add_argument("--keep", type=int, required=True, help="newest tags to keep per repo")
    parser.add_argument("--node", required=True, help="kind node container name")
    parser.add_argument("--registry-container", required=True)
    parser.add_argument("--kube-context", required=True, help="cluster whose pods pin tags")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    in_use = pods_image_refs(args.kube_context)
    kept = prune_registry(args, in_use)
    prune_node(args, kept, in_use)
    prune_host(args, kept, in_use)


if __name__ == "__main__":
    main()
