"""20 Newsgroups preprocessing: clean text, stratified splits, JSON export.

This module fetches the full 20 Newsgroups corpus, applies stopword removal and
Porter stemming, reports WordPiece OOV token counts (relative to DistilBERT's
vocabulary) without discarding any tokens, and writes stratified train/val/test
JSON files for downstream training.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Project paths (repo root: parent of src/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports" / "metrics"
DISTILBERT_VOCAB_MODEL = "distilbert-base-uncased"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StratifiedSplitIndex:
    """Holds array indices for train, validation, and test partitions."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def ensure_nltk_data() -> None:
    """Download required NLTK corpora and tokenizers if they are missing.

    Re-running downloads is a no-op when resources already exist. Uses
    ``quiet=True`` to reduce console noise.
    """
    import nltk

    for name in ("stopwords", "punkt", "punkt_tab"):
        nltk.download(name, quiet=True)


def get_stopwords() -> set[str]:
    """Return a set of English stopwords (lowercased) from NLTK."""
    from nltk.corpus import stopwords

    return {w.lower() for w in stopwords.words("english")}


def get_stemmer() -> Any:
    """Return a Porter stemmer for reuse across many documents."""
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def pretokenize_words(text: str) -> list[str]:
    """Split *text* into lowercased alphanumeric word tokens (regex-based).

    Args:
        text: Source document.

    Returns:
        List of lowercased token strings.
    """
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def preprocess_document(text: str, stop: set[str], stemmer: Any) -> str:
    """Remove stopwords, apply Porter stemming, and join to a single string.

    We do not drop OOV or rare tokens: everything that survives the above
    filters is kept for the subword tokenizer.

    Args:
        text: Source document.
        stop: Set of stopwords to drop.
        stemmer: An ``nltk.stem.porter.PorterStemmer`` instance.

    Returns:
        Space-separated, preprocessed text.
    """
    toks: list[str] = []
    for w in pretokenize_words(text):
        if w in stop:
            continue
        toks.append(stemmer.stem(w))
    return " ".join(toks)


def make_stratified_80_10_10(
    indices: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> StratifiedSplitIndex:
    """Stratify *y* and split *indices* into 80% / 10% / 10% partitions.

    Args:
        indices: Document indices (e.g. ``np.arange(n)`` for sklearn order).
        y: Integer labels, same length as *indices*.
        random_state: Seed for :class:`sklearn.model_selection.train_test_split`.

    Returns:
        Disjoint train, validation, and test index arrays.
    """
    if len(indices) != len(y):
        raise ValueError("indices and y must have the same length")
    if len(np.unique(y)) < 2:
        raise ValueError("stratify requires at least two distinct classes")

    tr_idx, temp_idx, _, y_temp = train_test_split(
        indices,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    v_idx, te_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return StratifiedSplitIndex(
        train=np.asarray(tr_idx, dtype=int),
        val=np.asarray(v_idx, dtype=int),
        test=np.asarray(te_idx, dtype=int),
    )


def count_oov_token_types_vs_wordpiece(
    documents: Sequence[str], model_name: str = DISTILBERT_VOCAB_MODEL
) -> Tuple[int, int]:
    """Count OOV *occurrences* and *unique types* w.r.t. the tokenizer vocab.

    A word token (after our preprocessing) is OOV if it is not a key in
    :meth:`transformers.PreTrainedTokenizer.get_vocab` and is not
    representable as ``##`` + *token* as a key. The model may still encode such
    strings via multiple subwords; we **never** remove them from ``text``.

    Args:
        documents: Preprocessed document strings.
        model_name: Hugging Face model id for that tokenizer.

    Returns:
        ``(n_oov_occurrences, n_unique_oov_types)``
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    keys = set(tok.get_vocab().keys())
    oov_types: set[str] = set()
    oov_occ = 0
    for doc in documents:
        for w in pretokenize_words(doc):
            if w in keys or f"##{w}" in keys:
                continue
            oov_occ += 1
            oov_types.add(w)
    return oov_occ, len(oov_types)


def _class_proportions(y: np.ndarray, n_class: int) -> np.ndarray:
    """Return per-class proportions for *y* in ``[0, n_class)`` order."""
    counts = np.bincount(y, minlength=n_class).astype(float)
    total = float(len(y))
    if total == 0:
        return np.zeros(n_class, dtype=float)
    return counts / total


def format_distribution(
    y: np.ndarray, target_names: list[str]
) -> str:
    """Format per-class counts and shares for console or log output.

    Args:
        y: Integer label vector.
        target_names: Human-readable class names, index-aligned.

    Returns:
        Multi-line string.
    """
    n = len(target_names)
    prop = _class_proportions(y, n)
    counts = np.bincount(y, minlength=n)
    lines: list[str] = []
    for i, name in enumerate(target_names):
        lines.append(
            f"  class {i:2d} {name!r:30s} count={counts[i]:5d}  p={prop[i]:.4f}"
        )
    return "\n".join(lines)


def _records_from_preprocessed(
    preprocessed: list[str],
    target: np.ndarray,
    target_names: list[str],
    indices: np.ndarray,
) -> list[dict[str, Any]]:
    """Build ``{text, label, label_name}`` rows for a split *indices*."""
    out: list[dict[str, Any]] = []
    for i in indices:
        t = preprocessed[i]
        li = int(target[i])
        out.append(
            {
                "text": t,
                "label": li,
                "label_name": target_names[li],
            }
        )
    return out


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run_preprocess(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    random_state: int = 42,
) -> None:
    """Fetch 20 Newsgroups, preprocess, log stats, and write JSON splits.

    Args:
        processed_dir: Directory for ``train.json``, ``validation.json``,
            ``test.json``.
        reports_dir: Directory for ``preprocess_distributions.log``.
        random_state: Random seed for stratified splitting.
    """
    ensure_nltk_data()
    logger.info("Fetching 20 Newsgroups (subset=all) ...")
    bunch = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=random_state,
    )
    raw = bunch.data
    y = np.asarray(bunch.target, dtype=int)
    target_names: list[str] = list(bunch.target_names)
    n = len(raw)
    indices = np.arange(n, dtype=int)
    n_class = len(target_names)

    full_prop = _class_proportions(y, n_class)
    logger.info("Class distribution (full corpus, n=%d)\n%s", n, format_distribution(y, target_names))

    stop = get_stopwords()
    stem = get_stemmer()

    split = make_stratified_80_10_10(indices, y, random_state=random_state)
    preprocessed: list[str] = [
        preprocess_document(raw[i], stop, stem) for i in range(n)
    ]
    oov_occ, oov_types = count_oov_token_types_vs_wordpiece(preprocessed)
    print(
        f"OOV vs DistilBERT WordPiece (after preprocess): "
        f"occurrences={oov_occ}, unique_types={oov_types}"
    )
    logger.info(
        "OOV (WordPiece) occurrences=%d unique_types=%d",
        oov_occ,
        oov_types,
    )

    train_rows = _records_from_preprocessed(
        preprocessed, y, target_names, split.train
    )
    val_rows = _records_from_preprocessed(
        preprocessed, y, target_names, split.val
    )
    test_rows = _records_from_preprocessed(
        preprocessed, y, target_names, split.test
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_json(processed_dir / "train.json", train_rows)
    _write_json(processed_dir / "validation.json", val_rows)
    _write_json(processed_dir / "test.json", test_rows)

    log_lines: list[str] = []
    for name, idx_arr, y_part in [
        ("train", split.train, y[split.train]),
        ("validation", split.val, y[split.val]),
        ("test", split.test, y[split.test]),
    ]:
        block = f"[{name}] n={len(y_part)}\n" + format_distribution(
            y_part, target_names
        )
        # Per-split proportion drift vs full
        p = _class_proportions(y_part, n_class)
        max_abs = float(np.max(np.abs(p - full_prop)))
        block += f"\n  max |p_split - p_full| (per class) = {max_abs:.4f}\n"
        log_lines.append(block)
        print(f"\n--- {name} (n={len(y_part)}) ---")
        print(block)

    reports_dir.mkdir(parents=True, exist_ok=True)
    dist_path = reports_dir / "preprocess_distributions.log"
    with dist_path.open("w", encoding="utf-8") as f:
        f.write("20 Newsgroups — per-split class distribution\n\n")
        f.write("\n\n".join(log_lines))
        f.write(
            f"\n\nOOV (DistilBERT WordPiece): "
            f"occurrences={oov_occ} unique_types={oov_types}\n"
        )
    logger.info("Wrote %s", dist_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the preprocessing entrypoint."""
    p = argparse.ArgumentParser(
        description="Preprocess 20 Newsgroups and write stratified JSON splits."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Output directory for train/validation/test JSON (default: data/processed).",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for preprocess_distributions.log (default: reports/metrics).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shuffling and stratified splits (default: 42).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging to stderr.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: configure logging and run :func:`run_preprocess`."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    run_preprocess(
        processed_dir=Path(args.out_dir),
        reports_dir=Path(args.report_dir),
        random_state=args.random_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
