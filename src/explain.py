"""Explain predictions and export deployment artifacts.

Capabilities:
1) SHAP explanations on selected test examples with LIME fallback if SHAP is
   too slow or unavailable.
2) ONNX export of the best checkpoint and parity verification on 3 examples.

All metrics/results are produced from actual runs. If no checkpoint is present,
this script writes explicit blocker artifacts in reports/metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import model as model_mod
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplainPaths:
    """All paths used by explain/export routines."""

    config: Path
    test_json: Path
    models_dir: Path
    figures_dir: Path
    metrics_dir: Path
    onnx_path: Path


class JsonDataset(Dataset):
    """JSON dataset with rows containing text, label, and label_name."""

    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"invalid dataset at {path}")
        self.rows: list[dict[str, Any]] = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config from disk."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_best_checkpoint(models_dir: Path) -> Path:
    """Locate the newest best (or epoch) checkpoint file."""
    best = sorted(models_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best:
        return best[0]
    ep = sorted(models_dir.glob("*_epoch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ep:
        return ep[0]
    raise FileNotFoundError("no checkpoint found in models/")


def get_device(force_cpu: bool = False) -> torch.device:
    """Select device for inference/export."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_blocker(paths: ExplainPaths, reason: str) -> None:
    """Write blocker files when explain/export cannot proceed."""
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "blocked",
        "reason": reason,
        "required_action": "Train model checkpoint in models/ then rerun src/explain.py",
    }
    (paths.metrics_dir / "explainability_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.error("Explainability/export blocked: %s", reason)


def _predict_proba_factory(
    net: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
) -> Any:
    """Return callable `texts -> probas` for SHAP/LIME."""

    def _predict(texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            enc = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            logits = net(ids, am)
            prob = torch.softmax(logits, dim=-1)
            return prob.detach().cpu().numpy()

    return _predict


def _pick_20_examples(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Select 5 examples from 4 random classes => 20 total."""
    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        c = int(r["label"])
        by_class.setdefault(c, []).append(r)
    classes = sorted(by_class.keys())
    rng.shuffle(classes)
    chosen_classes = classes[:4]
    out: list[dict[str, Any]] = []
    for c in chosen_classes:
        pool = by_class[c]
        rng.shuffle(pool)
        out.extend(pool[:5])
    return out[:20]


def run_explainability(
    net: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    out_dir: Path,
    max_length: int,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    """Run SHAP on 20 examples; fallback to LIME if SHAP exceeds timeout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = _pick_20_examples(rows)
    texts = [str(r["text"]) for r in picked]
    labels = [int(r["label"]) for r in picked]

    device = next(net.parameters()).device
    predict_fn = _predict_proba_factory(net, tokenizer, device, max_length)

    started = time.perf_counter()
    used = "shap"
    files: list[str] = []

    try:
        import shap

        explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text(tokenizer))
        shap_values = explainer(texts)
        elapsed = time.perf_counter() - started
        if elapsed > timeout_sec:
            raise TimeoutError(f"SHAP exceeded timeout {timeout_sec}s")

        # Save one summary bar plot and first 5 text plots for compactness.
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        bar_path = out_dir / "shap_summary_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=180)
        plt.close()
        files.append(str(bar_path))

        for i in range(min(5, len(texts))):
            plt.figure()
            shap.plots.text(shap_values[i], display=False)
            # text plot renders in notebook-like HTML; fallback: token importance bar
            # Extract absolute contribution per token for a static PNG
            vals = np.abs(np.array(shap_values.values[i])).sum(axis=-1)
            toks = shap_values.data[i]
            top = np.argsort(vals)[-20:]
            plt.figure(figsize=(10, 5))
            plt.barh(range(len(top)), vals[top])
            plt.yticks(range(len(top)), [str(toks[j]) for j in top])
            plt.title(f"SHAP token contributions example {i+1}")
            plt.tight_layout()
            fp = out_dir / f"shap_example_{i+1}.png"
            plt.savefig(fp, dpi=180)
            plt.close()
            files.append(str(fp))

    except Exception as exc:  # noqa: BLE001
        used = "lime"
        logger.warning("Falling back to LIME: %s", exc)
        from lime.lime_text import LimeTextExplainer

        class_names = sorted({str(r["label_name"]) for r in rows})
        explainer = LimeTextExplainer(class_names=class_names)
        for i, text in enumerate(texts[:20]):
            exp = explainer.explain_instance(text, predict_fn, num_features=12, top_labels=1)
            fig = exp.as_pyplot_figure()
            fig.tight_layout()
            fp = out_dir / f"lime_example_{i+1}.png"
            fig.savefig(fp, dpi=180)
            plt.close(fig)
            files.append(str(fp))

    return {
        "method": used,
        "n_examples": len(texts),
        "labels": labels,
        "figure_files": files,
    }


def export_onnx_and_verify(
    net: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    onnx_path: Path,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    """Export model to ONNX and verify parity on 3 test examples."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    net_cpu = net.to("cpu").eval()

    sample = rows[:3]
    texts = [str(r["text"]) for r in sample]
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ids = enc["input_ids"]
    am = enc["attention_mask"]

    torch.onnx.export(
        net_cpu,
        (ids, am),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify parity with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = sess.run(
        ["logits"],
        {
            "input_ids": ids.numpy().astype(np.int64),
            "attention_mask": am.numpy().astype(np.int64),
        },
    )[0]

    with torch.inference_mode():
        pt_logits = net_cpu(ids, am).numpy()

    pt_pred = np.argmax(pt_logits, axis=-1)
    ort_pred = np.argmax(ort_logits, axis=-1)
    pred_match = bool(np.array_equal(pt_pred, ort_pred))

    return {
        "onnx_path": str(onnx_path),
        "checked_examples": len(texts),
        "pytorch_preds": pt_pred.tolist(),
        "onnx_preds": ort_pred.tolist(),
        "prediction_match": pred_match,
    }


def run(paths: ExplainPaths, max_length: int, force_cpu: bool) -> int:
    """Execute explainability and ONNX export/verification."""
    cfg = load_config(paths.config)
    try:
        ckpt = find_best_checkpoint(paths.models_dir)
    except FileNotFoundError as exc:
        write_blocker(paths, str(exc))
        return 2

    payload = torch.load(ckpt, map_location="cpu")
    model_name = str(payload.get("model_name", cfg.get("model_name", "distilbert-base-uncased")))
    dropout = float(payload.get("dropout", cfg.get("dropout", 0.1)))
    num_labels = int(cfg.get("num_labels", 20))

    with paths.test_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    net = model_mod.DistilBertForNewsgroups(model_name, num_labels=num_labels, dropout=dropout)
    net.load_state_dict(payload["model_state_dict"], strict=True)
    device = get_device(force_cpu=force_cpu)
    net.to(device).eval()

    explain_report = run_explainability(
        net,
        tokenizer,
        rows,
        paths.figures_dir,
        max_length=max_length,
        timeout_sec=300,
    )

    onnx_report = export_onnx_and_verify(
        net,
        tokenizer,
        rows,
        paths.onnx_path,
        max_length=max_length,
        device=device,
    )

    payload_out = {
        "status": "ok",
        "checkpoint": str(ckpt),
        "device": str(device),
        "explainability": explain_report,
        "onnx": onnx_report,
    }
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    (paths.metrics_dir / "explainability_report.json").write_text(
        json.dumps(payload_out, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload_out, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI parser."""
    p = argparse.ArgumentParser(description="Explain predictions and export ONNX.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "training_config.yaml")
    p.add_argument("--test-json", type=Path, default=REPO_ROOT / "data" / "processed" / "test.json")
    p.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    p.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "reports" / "figures")
    p.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "reports" / "metrics")
    p.add_argument("--onnx-path", type=Path, default=REPO_ROOT / "models" / "model.onnx")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--cpu", action="store_true", help="Force CPU for runtime.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Program entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = ExplainPaths(
        config=args.config if args.config.is_absolute() else REPO_ROOT / args.config,
        test_json=args.test_json if args.test_json.is_absolute() else REPO_ROOT / args.test_json,
        models_dir=args.models_dir if args.models_dir.is_absolute() else REPO_ROOT / args.models_dir,
        figures_dir=args.figures_dir if args.figures_dir.is_absolute() else REPO_ROOT / args.figures_dir,
        metrics_dir=args.metrics_dir if args.metrics_dir.is_absolute() else REPO_ROOT / args.metrics_dir,
        onnx_path=args.onnx_path if args.onnx_path.is_absolute() else REPO_ROOT / args.onnx_path,
    )
    return run(paths, max_length=int(args.max_length), force_cpu=bool(args.cpu))


if __name__ == "__main__":
    raise SystemExit(main())
