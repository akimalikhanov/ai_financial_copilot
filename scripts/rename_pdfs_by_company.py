#!/usr/bin/env python3
"""Rename PDFs in data/pdfs from sha1.pdf to company_name.pdf using subset.csv mapping."""

import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDFS_DIR = DATA_DIR / "pdfs"
CSV_PATH = DATA_DIR / "subset.csv"


def sanitize_filename(name: str) -> str:
    """Replace characters invalid in filenames with underscore."""
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip()


def main() -> None:
    # Build sha1 -> company_name mapping
    mapping: dict[str, str] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sha1 = row["sha1"].strip()
            company_name = row["company_name"].strip()
            mapping[sha1] = company_name

    # Track used names to handle duplicates
    used_names: dict[str, int] = {}

    for sha1, company_name in mapping.items():
        src = PDFS_DIR / f"{sha1}.pdf"
        if not src.exists():
            print(f"Skip (not found): {src.name}")
            continue

        base_name = sanitize_filename(company_name)
        if base_name in used_names:
            used_names[base_name] += 1
            new_name = f"{base_name} ({used_names[base_name]}).pdf"
        else:
            used_names[base_name] = 1
            new_name = f"{base_name}.pdf"

        dst = PDFS_DIR / new_name
        if dst == src:
            continue
        if dst.exists():
            print(f"Skip (target exists): {src.name} -> {new_name}")
            continue

        src.rename(dst)
        print(f"Renamed: {src.name} -> {new_name}")


if __name__ == "__main__":
    main()
