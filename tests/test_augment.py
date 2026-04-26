"""Light tests for *augment* (no-augment pass-through and shape)."""

from __future__ import annotations

import json
from pathlib import Path

import augment


def test_augment_pass_through_preserves_length(tmp_path: Path) -> None:
    """With ``do_augment`` False, output matches input row count and keys."""
    inp = tmp_path / "t.json"
    data = [
        {"text": "cat dog", "label": 0, "label_name": "a"},
        {"text": "x y z", "label": 1, "label_name": "b"},
    ]
    with inp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    out = tmp_path / "o.json"
    augment.run_augment(inp, out, do_augment=False, seed=0, p_token=0.1)
    with out.open("r", encoding="utf-8") as f:
        out_data = json.load(f)
    assert out_data == data
