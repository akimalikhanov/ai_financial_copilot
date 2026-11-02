"""Compare two rendered manifest streams object by object, ignoring Kustomize/Helm noise.

Kustomize hash suffixes, Helm release labels, and checksum/helm.sh annotations are removed first.
Secrets are dropped so no credential is ever printed.
"""

import argparse
import difflib
import re
import sys
from typing import Any

import yaml

HELM_LABELS = (
    "helm.sh/chart",
    "app.kubernetes.io/managed-by",
    "app.kubernetes.io/instance",
    "app.kubernetes.io/version",
)
# Kustomize's 10-char suffix alphabet: hex with 0,1,3,a,e swapped for g,h,k,m,t.
HASH_SUFFIX = re.compile(r"^(?P<base>.+)-[2456789bcdfghkmt]{10}$")


def load(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [d for d in yaml.safe_load_all(f) if d]


def unhash(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renames = {}
    for d in docs:
        if d["kind"] in ("ConfigMap", "Secret"):
            m = HASH_SUFFIX.match(d["metadata"]["name"])
            if m:
                renames[d["metadata"]["name"]] = m["base"]
    if not renames:
        return docs
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, renames)) + r")\b")
    text = pattern.sub(lambda m: renames[m[1]], yaml.safe_dump_all(docs))
    return [d for d in yaml.safe_load_all(text) if d]


def scrub(meta: dict[str, Any] | None) -> None:
    if not meta:
        return
    for k in HELM_LABELS:
        (meta.get("labels") or {}).pop(k, None)
    annotations = meta.get("annotations") or {}
    for k in [k for k in annotations if k.startswith(("checksum/", "helm.sh/"))]:
        del annotations[k]
    for key in ("labels", "annotations"):
        if key in meta and not meta[key]:
            del meta[key]


def index(docs: list[dict[str, Any]], match: str | None) -> dict[str, list[str]]:
    out = {}
    for d in docs:
        name = d["metadata"]["name"]
        if d["kind"] == "Secret" or (match and not re.search(match, name)):
            continue
        scrub(d["metadata"])
        scrub((d.get("spec") or {}).get("template", {}).get("metadata"))
        out[f"{d['kind']}/{name}"] = yaml.safe_dump(d, sort_keys=True).splitlines()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("--names", nargs=2, default=("left", "right"))
    p.add_argument("--match", help="only compare objects whose name matches this regex")
    args = p.parse_args()

    left = index(unhash(load(args.left)), args.match)
    right = index(unhash(load(args.right)), args.match)
    changed = False
    for key in sorted(left.keys() | right.keys()):
        if key not in right:
            print(f"only in {args.names[0]}: {key}")
        elif key not in left:
            print(f"only in {args.names[1]}: {key}")
        else:
            diff = list(
                difflib.unified_diff(
                    left[key],
                    right[key],
                    lineterm="",
                    fromfile=f"{args.names[0]}/{key}",
                    tofile=f"{args.names[1]}/{key}",
                )
            )
            if not diff:
                continue
            print("\n".join(diff))
        changed = True
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
