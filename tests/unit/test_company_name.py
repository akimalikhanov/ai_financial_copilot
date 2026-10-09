"""Snapshot of normalize_company on real stored names.

A failure here means stored `company_norm` values are stale: update the snapshot, then
rerun `.venv/bin/python -m scripts.backfill_company_norm`.
"""

from __future__ import annotations

import pytest

from src.services.router.company_name import normalize_company

SNAPSHOT = [
    ("Aurora Innovation, Inc.", "aurora innovation"),
    ("Microsoft Corporation (scanned 10p)", "microsoft"),
    ("Poste Italiane S.p.A.", "poste italiane"),
    ("Mosaic Brands Limited", "mosaic brands"),
    ("AA Limited", "aa"),
    ("the AA", "the aa"),
    ("Blue Apron Holdings, Inc.", "blue apron"),
    ("Elixir Energy Limited", "elixir energy"),
    ("Benjamin Hornigold Ltd", "benjamin hornigold"),
    ("Pashabank", "pashabank"),
    ("RWE AG", "rwe"),
    ("Alcoa Corporation", "alcoa"),
    ("Unibank", "unibank"),
    ("Capital One Financial", "capital one financial"),
    ("Société Générale S.A.", "societe generale"),
    ("AT&T Inc.", "at t"),
    ("Holdings Ltd", "holdings"),
    ("  aurora  ", "aurora"),
]


@pytest.mark.parametrize(("name", "expected"), SNAPSHOT)
def test_normalize_company_snapshot(name: str, expected: str) -> None:
    assert normalize_company(name) == expected
