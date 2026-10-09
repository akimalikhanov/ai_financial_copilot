"""Materialise item 2.2's 20-document backlog from the real corpus.

Reads infra/loadtest/backlog_sample.txt — which records the sampling decision and why — and
copies each named filing out of the corpus into the fixtures directory under a shell-safe name.

Separate from make_fixtures.py on purpose. That script *derives* fixtures from one source
document (truncating, rasterizing, synthesizing); this one only selects and copies files that
already exist. Mixing them would put a corpus dependency into the fixture generator.

Files land in a `backlog/` subdirectory rather than alongside normal.pdf, so a run selects the
scenario with LOADTEST_FIXTURE_DIR instead of listing twenty names in LOADTEST_FIXTURES. Names
are rewritten to `NN-slug.pdf` for two reasons: the corpus contains filenames with commas
("1-800-FLOWERS.COM, INC..pdf"), which would break LOADTEST_FIXTURES' comma-separated parsing if
anyone later pinned one; and the NN prefix preserves the manifest's ordering, so a document in a
log or CSV is traceable back to its stratum without consulting the database.

Usage:
    .venv/bin/python -m infra.loadtest.make_backlog
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

DEFAULT_MANIFEST = Path("infra/loadtest/backlog_sample.txt")
DEFAULT_CORPUS = Path("data/corpus/pdfs")
DEFAULT_OUT = Path("infra/k8s/loadtest/fixtures/backlog")


def _slug(filename: str) -> str:
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return stem or "document"


def read_manifest(path: Path) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, pages, seconds = line.split("|")
        rows.append((name, int(pages), float(seconds)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    missing = [n for n, _, _ in rows if not (args.corpus / n).is_file()]
    if missing:
        # Abort rather than run short: a backlog of 17 is not the backlog the queue curve is
        # being read against, and nothing downstream would say so.
        raise SystemExit(f"{len(missing)} file(s) missing from {args.corpus}: {missing}")

    # Rebuild from empty: a stale file from an earlier sample would silently join the pool,
    # since a run uploads everything in the directory.
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    total = 0
    print()
    for i, (name, pages, seconds) in enumerate(rows, start=1):
        dest = args.out / f"{i:02d}-{_slug(name)}.pdf"
        shutil.copy2(args.corpus / name, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        total += dest.stat().st_size
        print(f"  {dest.name:44} {pages:5d}p {size_mb:6.1f} MB   was {seconds:7.1f}s")

    print(f"\n{len(rows)} document(s), {total / (1024 * 1024):.0f} MB total -> {args.out}")
    print("service times above are the OLD pipeline — the stratification key, not a prediction")


if __name__ == "__main__":
    main()
