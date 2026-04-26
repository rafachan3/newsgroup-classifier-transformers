"""Tests for 20 Newsgroup stratified splits: leakage and class balance.

These tests use the public sklearn load (cached after the first run) and the
same split contract as :mod:`preprocess` (``random_state=42`` and shuffled
fetch). We never touch held-out *test* labels for any modeling decision; these
are pure split integrity checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import fetch_20newsgroups

import preprocess

TOL = 0.02


@pytest.fixture(scope="module")
def loaded_20ng() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load full 20 Newsgroups with the same options as *preprocess*."""
    bunch = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    n = len(bunch.data)
    indices = np.arange(n, dtype=int)
    y = np.asarray(bunch.target, dtype=int)
    names = list(bunch.target_names)
    return indices, y, names


def test_no_index_overlap_across_splits(loaded_20ng) -> None:
    """Train, val, and test index sets are pairwise disjoint and complete."""
    indices, y, _ = loaded_20ng
    split = preprocess.make_stratified_80_10_10(
        indices, y, random_state=42
    )
    s_tr, s_v, s_te = set(split.train), set(split.val), set(split.test)
    assert s_tr & s_v == set()
    assert s_tr & s_te == set()
    assert s_v & s_te == set()
    assert s_tr | s_v | s_te == set(range(len(y)))


def test_class_proportions_within_tolerance(loaded_20ng) -> None:
    """Each split’s per-class share is within *TOL* of the full corpus."""
    indices, y, _ = loaded_20ng
    n_class = int(np.max(y)) + 1
    full = preprocess._class_proportions(y, n_class)
    split = preprocess.make_stratified_80_10_10(
        indices, y, random_state=42
    )
    for part_name, idx in (
        ("train", split.train),
        ("validation", split.val),
        ("test", split.test),
    ):
        p = preprocess._class_proportions(y[idx], n_class)
        max_delta = float(np.max(np.abs(p - full)))
        assert max_delta < TOL, f"{part_name} max |p - p_full| = {max_delta} >= {TOL}"


def test_output_json_has_required_fields(tmp_path: Path) -> None:
    """Smoke: running *run_preprocess* writes three JSON files with 3 fields."""
    out = tmp_path / "processed"
    rep = tmp_path / "rep"
    preprocess.run_preprocess(processed_dir=out, reports_dir=rep, random_state=42)
    for name in ("train.json", "validation.json", "test.json"):
        with (out / name).open("r", encoding="utf-8") as f:
            rows = json.load(f)
        assert isinstance(rows, list) and len(rows) > 0
        for r in rows:
            assert set(r.keys()) == {"text", "label", "label_name"}
            assert isinstance(r["text"], str)
            assert isinstance(r["label"], int)
            assert isinstance(r["label_name"], str)
