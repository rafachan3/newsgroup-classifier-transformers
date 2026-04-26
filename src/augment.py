"""Optional training-set augmentation via WordNet synonym replacement.

This module reads a *training-only* JSON file produced by
:mod:`preprocess` and, when ``--augment`` is set, rewrites a subset of words
using synonym substitution. Validation and test sets must not be passed through
this script in the intended pipeline.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "train.json"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "train_aug.json"


def ensure_nltk_wordnet() -> None:
    """Download WordNet and multilingual Wordnet data if needed."""
    import nltk

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def replace_with_synonym(
    word: str, rng: random.Random, p_replace: float
) -> str:
    """Stochastically replace *word* with a WordNet lemma name, or return *word*.

    Args:
        word: A single preprocessed token (alphanumeric, lowercased).
        rng: Random source.
        p_replace: Probability of attempting a replacement (not guaranteed).

    Returns:
        Either a synonym or the original *word* if none applies.
    """
    if not word or rng.random() > p_replace:
        return word
    from nltk.corpus import wordnet as wn

    syns = wn.synsets(word)
    if not syns:
        return word
    s = syns[rng.randrange(len(syns))]
    lemmas = s.lemmas()
    if not lemmas:
        return word
    candidate = lemmas[rng.randrange(len(lemmas))].name()
    return candidate.replace("_", " ")


def augment_text(text: str, rng: random.Random, p_token: float) -> str:
    """Apply token-wise synonym replacement to a single document string.

    Args:
        text: Space-separated tokens (as in ``preprocess`` output).
        rng: Random source.
        p_token: Per-token attempt probability (see :func:`replace_with_synonym`).

    Returns:
        Augmented string.
    """
    words = text.split()
    if not words:
        return text
    return " ".join(
        replace_with_synonym(w, rng, p_token) for w in words
    )


def load_train_rows(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of training examples from *path*."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of training examples")
    return data


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write *rows* to *path* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_augment(
    in_path: Path,
    out_path: Path,
    *,
    do_augment: bool,
    seed: int = 42,
    p_token: float = 0.1,
) -> None:
    """Read training JSON, optionally augment, and write to *out_path*.

    Args:
        in_path: Input ``train.json`` (or path produced by *preprocess*).
        out_path: Output path (e.g. ``train_aug.json`` or overwrite path).
        do_augment: If False, a deep-copied list is written unchanged.
        seed: RNG seed (used only when *do_augment* is True).
        p_token: Per-token replacement attempt probability.
    """
    rows = load_train_rows(in_path)
    if not do_augment:
        out_rows = copy.deepcopy(rows)
    else:
        ensure_nltk_wordnet()
        rng = random.Random(seed)
        out_rows = []
        for r in rows:
            rec = copy.deepcopy(r)
            rec["text"] = augment_text(
                str(rec.get("text", "")),
                rng,
                p_token,
            )
            out_rows.append(rec)
    save_rows(out_path, out_rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Argument parser for :func:`main`."""
    p = argparse.ArgumentParser(
        description="Optional WordNet synonym augmentation for train.json only."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TRAIN,
        help="Path to input training JSON (default: data/processed/train.json).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to output JSON (default: data/processed/train_aug.json).",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="If set, apply synonym replacement. Otherwise the file is copied.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    p.add_argument(
        "--p-token",
        type=float,
        default=0.1,
        help="Per-token attempt probability (default: 0.1).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Script entry: parse args and run :func:`run_augment`."""
    args = parse_args(argv)
    run_augment(
        Path(args.input),
        Path(args.output),
        do_augment=bool(args.augment),
        seed=int(args.seed),
        p_token=float(args.p_token),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
