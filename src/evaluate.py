"""Evaluate a trained DistilBERT checkpoint on the held-out test split.

Outputs:
- reports/metrics/test_metrics.json
- reports/figures/confusion_matrix.png
- reports/metrics/error_analysis.json
- reports/figures/attention_*.png
- reports/metrics/bias_analysis.md

If no checkpoint is available, the script writes blocker files rather than
fabricating metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import model as model_mod
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalPaths:
    """Resolved IO paths for evaluation outputs."""

    test_json: Path
    model_dir: Path
    fig_dir: Path
    metrics_dir: Path
    config_path: Path


class JsonDataset(Dataset):
    """JSON dataset with rows: text,label,label_name."""

    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"invalid or empty dataset: {path}")
        self.rows: list[dict[str, Any]] = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_collate(tokenizer: Any, max_length: int) -> Any:
    """Return a collate_fn that tokenizes text and keeps metadata."""

    def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(x["text"]) for x in batch]
        labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
        label_names = [str(x["label_name"]) for x in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "texts": texts,
            "labels": labels,
            "label_names": label_names,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    return _collate


def get_device(force_cpu: bool = False) -> torch.device:
    """Select runtime device."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_best_checkpoint(model_dir: Path) -> Path:
    """Find best checkpoint file in model_dir.

    Preference order:
    1) *_best.pt files (newest first)
    2) *_epoch*.pt files (newest first)
    """
    best = sorted(model_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    epochs = sorted(model_dir.glob("*_epoch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if epochs:
        return epochs[0]
    raise FileNotFoundError("no checkpoint found in models/")


def write_blocker_artifacts(paths: EvalPaths, reason: str) -> None:
    """Emit explicit blocker files when evaluation cannot run."""
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.fig_dir.mkdir(parents=True, exist_ok=True)

    blocker_json = {
        "status": "blocked",
        "reason": reason,
        "required_action": "Train model to produce checkpoint in models/ then rerun evaluate.py",
    }
    (paths.metrics_dir / "error_analysis.json").write_text(
        json.dumps(blocker_json, indent=2) + "\n",
        encoding="utf-8",
    )
    (paths.metrics_dir / "test_metrics.json").write_text(
        json.dumps(blocker_json, indent=2) + "\n",
        encoding="utf-8",
    )
    (paths.metrics_dir / "bias_analysis.md").write_text(
        "# Bias analysis\n\n"
        "Evaluation blocked.\n\n"
        f"Reason: {reason}\n",
        encoding="utf-8",
    )


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out: Path) -> None:
    """Save confusion matrix heatmap."""
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("20 Newsgroups Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def top_confused_pairs(cm: np.ndarray, labels: list[str], k: int = 3) -> list[dict[str, Any]]:
    """Return top-k off-diagonal confusion pairs with short analysis text."""
    pairs: list[tuple[int, int, int]] = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    out: list[dict[str, Any]] = []
    for i, j, c in pairs[:k]:
        out.append(
            {
                "true_class": labels[i],
                "pred_class": labels[j],
                "count": c,
                "analysis": (
                    f"{labels[i]} is often mistaken for {labels[j]} ({c} times). "
                    "This likely reflects lexical overlap after stopword removal and stemming. "
                    "Context-bearing tokens may be truncated or too sparse for robust separation."
                ),
            }
        )
    return out


def save_attention_heatmap(
    weights: np.ndarray,
    tokens: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Save a token-token attention map."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(tokens), 40)
    plt.figure(figsize=(12, 10))
    sns.heatmap(weights[:n, :n], cmap="magma")
    plt.title(title)
    plt.xlabel("Token")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def eval_and_collect(
    net: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
) -> dict[str, Any]:
    """Run inference and collect predictions + attention examples."""
    net.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    label_name_by_id: dict[int, str] = {}

    attention_correct: list[dict[str, Any]] = []
    attention_wrong: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = net.bert(
                input_ids=ids,
                attention_mask=am,
                output_attentions=True,
                return_dict=True,
            )
            pooled = out.last_hidden_state[:, 0, :]
            logits = net.classifier(net.dropout(pooled))
            pred = torch.argmax(logits, dim=-1)

            yt = labels.cpu().numpy().tolist()
            yp = pred.cpu().numpy().tolist()
            y_true.extend(yt)
            y_pred.extend(yp)

            # last layer attentions: (batch, heads, seq, seq)
            last_attn = out.attentions[-1].detach().cpu().numpy()
            ids_np = ids.detach().cpu().numpy()
            am_np = am.detach().cpu().numpy()

            for i in range(len(yt)):
                tid = int(yt[i])
                if tid not in label_name_by_id:
                    label_name_by_id[tid] = batch["label_names"][i]

                seq_len = int(np.sum(am_np[i]))
                attn = np.mean(last_attn[i, :, :seq_len, :seq_len], axis=0)
                tok = tokenizer.convert_ids_to_tokens(ids_np[i][:seq_len].tolist())
                rec = {
                    "tokens": tok,
                    "attn": attn,
                    "true": int(yt[i]),
                    "pred": int(yp[i]),
                }
                if yt[i] == yp[i] and len(attention_correct) < 5:
                    attention_correct.append(rec)
                elif yt[i] != yp[i] and len(attention_wrong) < 5:
                    attention_wrong.append(rec)

    return {
        "y_true": np.array(y_true, dtype=int),
        "y_pred": np.array(y_pred, dtype=int),
        "label_name_by_id": label_name_by_id,
        "attention_correct": attention_correct,
        "attention_wrong": attention_wrong,
    }


def write_bias_analysis(report: dict[str, Any], out_path: Path) -> None:
    """Write class-level bias/performance notes for classes with F1 < 0.60."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Bias analysis", ""]
    per_class = report["per_class"]
    low = [x for x in per_class if float(x["f1"]) < 0.60]
    if not low:
        lines.append("No class has F1 below 0.60 in this run.")
    else:
        lines.append("Classes below F1=0.60 and likely causes:")
        lines.append("")
        for item in low:
            lines.append(
                f"- {item['label_name']} (F1={item['f1']:.3f}, support={item['support']}): "
                "likely topic overlap with neighboring classes and sparse topical cues "
                "after aggressive normalization."
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(paths: EvalPaths, batch_size: int, max_length: int, force_cpu: bool = False) -> int:
    """Load checkpoint, evaluate test split, and write all Phase-4 artifacts."""
    try:
        ckpt = find_best_checkpoint(paths.model_dir)
    except FileNotFoundError as exc:
        reason = str(exc)
        write_blocker_artifacts(paths, reason)
        logger.error("Evaluation blocked: %s", reason)
        return 2

    cfg = load_config(paths.config_path)
    model_name = str(cfg["model_name"])
    num_labels = int(cfg.get("num_labels", 20))

    payload = torch.load(ckpt, map_location="cpu")
    dropout = float(payload.get("dropout", cfg.get("dropout", 0.1)))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = JsonDataset(paths.test_json)
    collate = make_collate(tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    net = model_mod.DistilBertForNewsgroups(model_name, num_labels=num_labels, dropout=dropout)
    net.load_state_dict(payload["model_state_dict"], strict=True)
    device = get_device(force_cpu=force_cpu)
    net.to(device)

    gathered = eval_and_collect(net, loader, device, tokenizer)
    y_true = gathered["y_true"]
    y_pred = gathered["y_pred"]

    label_name_by_id = gathered["label_name_by_id"]
    ids = sorted(label_name_by_id.keys())
    labels = [label_name_by_id[i] for i in ids]

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    # per-class using sorted ids
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    per_class = []
    for idx, cls_id in enumerate(ids):
        per_class.append(
            {
                "label_id": int(cls_id),
                "label_name": labels[idx],
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f1[idx]),
                "support": int(s[idx]),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=ids)
    plot_confusion_matrix(cm, labels, paths.fig_dir / "confusion_matrix.png")

    confused = top_confused_pairs(cm, labels, k=3)

    # Attention figures
    fig_idx = 0
    for rec in gathered["attention_correct"]:
        fig_idx += 1
        save_attention_heatmap(
            rec["attn"],
            rec["tokens"],
            paths.fig_dir / f"attention_correct_{fig_idx}.png",
            title=f"Correct example {fig_idx} (true={rec['true']}, pred={rec['pred']})",
        )
    fig_idx = 0
    for rec in gathered["attention_wrong"]:
        fig_idx += 1
        save_attention_heatmap(
            rec["attn"],
            rec["tokens"],
            paths.fig_dir / f"attention_misclassified_{fig_idx}.png",
            title=f"Misclassified example {fig_idx} (true={rec['true']}, pred={rec['pred']})",
        )

    report_payload = {
        "checkpoint": str(ckpt),
        "device": str(device),
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
        },
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=ids,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    (paths.metrics_dir / "test_metrics.json").write_text(
        json.dumps(report_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (paths.metrics_dir / "error_analysis.json").write_text(
        json.dumps(
            {
                "top_confused_pairs": confused,
                "summary": "Top off-diagonal confusions sorted by count.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_bias_analysis(report_payload, paths.metrics_dir / "bias_analysis.md")

    logger.info(
        "Test macro F1=%.4f, micro F1=%.4f", float(macro_f1), float(micro_f1)
    )
    print(
        json.dumps(
            {
                "macro_f1": float(macro_f1),
                "micro_f1": float(micro_f1),
                "checkpoint": str(ckpt),
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(description="Evaluate best checkpoint on test split.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "training_config.yaml")
    p.add_argument("--test-json", type=Path, default=REPO_ROOT / "data" / "processed" / "test.json")
    p.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    p.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "reports" / "metrics")
    p.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "reports" / "figures")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = EvalPaths(
        test_json=args.test_json if args.test_json.is_absolute() else REPO_ROOT / args.test_json,
        model_dir=args.models_dir if args.models_dir.is_absolute() else REPO_ROOT / args.models_dir,
        fig_dir=args.figures_dir if args.figures_dir.is_absolute() else REPO_ROOT / args.figures_dir,
        metrics_dir=args.metrics_dir if args.metrics_dir.is_absolute() else REPO_ROOT / args.metrics_dir,
        config_path=args.config if args.config.is_absolute() else REPO_ROOT / args.config,
    )
    return run(paths, batch_size=int(args.batch_size), max_length=int(args.max_length), force_cpu=bool(args.cpu))


if __name__ == "__main__":
    raise SystemExit(main())
