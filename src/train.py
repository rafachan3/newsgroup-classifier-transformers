"""Fine-tune DistilBERT for 20 Newsgroup classification (AdamW, warmup, F1, ES)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import model as model_mod
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Fix RNGs for Python, NumPy, and torch (as far as possible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(*, force_cpu: bool = False) -> torch.device:
    """Prefer CUDA, then MPS, then CPU—unless *force_cpu* is set.

    On some Apple Silicon setups, MPS can OOM during AdamW updates; use
    ``--cpu`` to fall back to CPU (slower but stable).
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config relative to the repo (or an absolute path)."""
    p = (REPO_ROOT / path) if not path.is_absolute() else path
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class JsonTextDataset(Dataset):
    """JSON list of ``{text, label, label_name}`` rows."""

    def __init__(self, path: Path) -> None:
        p = (REPO_ROOT / path) if not path.is_absolute() else path
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.rows: list[dict[str, Any]] = raw
        if not self.rows:
            raise ValueError(f"empty dataset: {p}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        r = self.rows[idx]
        return str(r["text"]), int(r["label"])


def make_collate_fn(
    tokenizer: Any, max_length: int
) -> Any:
    """Build a collate that batch-tokenizes *text* strings."""

    def _collate(batch: list[tuple[str, int]]) -> dict[str, Any]:
        texts = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return _collate


def _evaluate_macro_f1(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_cuda_amp: bool,
) -> float:
    """Compute validation macro F1 (primary model-selection metric)."""
    net.eval()
    ys: list[np.ndarray] = []
    yps: list[np.ndarray] = []
    nb = bool(device.type == "cuda")
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=nb)
            am = batch["attention_mask"].to(device, non_blocking=nb)
            lab = batch["labels"].to(device, non_blocking=nb)
            with autocast(enabled=use_cuda_amp):
                logits = net(input_ids, am)
            pred = torch.argmax(logits, dim=-1)
            ys.append(lab.cpu().numpy())
            yps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(yps, axis=0)
    return float(f1_score(y_true, y_pred, average="macro"))


def _train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: GradScaler,
    use_cuda_amp: bool,
    max_norm: float,
) -> float:
    """Run one pass over *loader*; return mean cross-entropy loss."""
    net.train()
    tot = 0.0
    n_samples = 0
    nb = bool(device.type == "cuda")
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=nb)
        am = batch["attention_mask"].to(device, non_blocking=nb)
        lab = batch["labels"].to(device, non_blocking=nb)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_cuda_amp):
            logits = net(input_ids, am)
            loss: torch.Tensor = criterion(logits, lab)
        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            optimizer.step()
        scheduler.step()
        bs = int(input_ids.size(0))
        tot += float(loss.item()) * bs
        n_samples += bs
        if device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return tot / max(n_samples, 1)


def append_training_log(
    log_path: Path,
    title: str,
    body_lines: Sequence[str],
) -> None:
    """Append a block to the markdown training log (create parent dirs)."""
    log_path = REPO_ROOT / log_path if not log_path.is_absolute() else log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    block = [f"## {title} ({stamp})\n", "\n", *[f"{line}\n" for line in body_lines], "\n"]
    with log_path.open("a", encoding="utf-8") as f:
        f.writelines(block)


def train_one_run(
    cfg: Mapping[str, Any],
    *,
    dropout: float,
    learning_rate: float,
    output_dir: Path,
    autoretry_lr: bool = True,
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Single training run. Returns a dict of summary metrics and paths."""
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(force_cpu=force_cpu)
    use_cuda_amp = bool(torch.cuda.is_available())
    if use_cuda_amp:
        logger.info("Using CUDA and mixed precision (autocast).")
    else:
        logger.info("Mixed precision (cuda.amp) disabled (no CUDA). device=%s", device)

    model_name = str(cfg["model_name"])
    num_labels = int(cfg["num_labels"])
    max_length = int(cfg["max_length"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    patience = int(cfg.get("early_stopping_patience", 2))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = JsonTextDataset(Path(cfg["train_path"]))
    val_ds = JsonTextDataset(Path(cfg["validation_path"]))
    collate = make_collate_fn(tokenizer, max_length)
    num_workers = 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    net = model_mod.DistilBertForNewsgroups(
        model_name, num_labels=num_labels, dropout=dropout
    )
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    params: list[dict[str, Any]] = [
        {
            "params": [
                p
                for n, p in net.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in net.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(params, lr=learning_rate, eps=1e-8)
    tsteps = int(len(train_loader) * max(1, epochs))
    warmup = int(tsteps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=tsteps
    )
    scaler = GradScaler(enabled=use_cuda_amp)

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    out_dir = REPO_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"dropout_{dropout}_lr_{learning_rate}".replace(".", "p")
    per_epoch: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss = _train_one_epoch(
            net,
            train_loader,
            device,
            criterion,
            opt,
            sched,
            scaler,
            use_cuda_amp,
            max_norm,
        )
        val_f1 = _evaluate_macro_f1(
            net, val_loader, device, use_cuda_amp
        )
        dt = time.perf_counter() - t0
        line = (
            f"Epoch {epoch}/{epochs}  train_loss={tr_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  ({dt:.0f}s)"
        )
        print(line, flush=True)
        logger.info(line)
        per_epoch.append(
            {
                "epoch": float(epoch),
                "train_loss": float(tr_loss),
                "val_macro_f1": float(val_f1),
            }
        )
        ckpt = out_dir / f"{run_tag}_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "val_macro_f1": val_f1,
                "dropout": dropout,
                "model_name": model_name,
            },
            ckpt,
        )
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            best_path = out_dir / f"{run_tag}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "val_macro_f1": val_f1,
                    "dropout": dropout,
                    "model_name": model_name,
                },
                best_path,
            )
        else:
            no_improve += 1
        if no_improve >= patience:
            logger.info("Early stop: no val F1 gain for %d epoch(s).", patience)
            break

    summary = {
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "per_epoch": per_epoch,
        "device": str(device),
    }

    if autoretry_lr and best_f1 < 0.7:
        logger.info(
            "Val macro F1 < 0.70; retrying once with 50%% learning rate (per spec)."
        )
        append_training_log(
            Path(cfg.get("log_path", "reports/training_log.md")),
            f"LR retry (dropout={dropout})",
            [
                f"First run best val macro F1: {best_f1:.4f} — running again with "
                f"learning_rate {learning_rate} -> {learning_rate * 0.5:.2e}",
            ],
        )
        return train_one_run(
            cfg,
            dropout=dropout,
            learning_rate=learning_rate * 0.5,
            output_dir=output_dir,
            autoretry_lr=False,
            force_cpu=force_cpu,
        )

    return summary


def run_from_args(args: argparse.Namespace) -> int:
    """Entry for CLI: load config, maybe sweep dropout, log results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cfg = load_config(Path(args.config))
    out_dir = Path(cfg.get("output_dir", "models"))
    d_out = float(cfg.get("dropout", 0.1))
    base_lr = float(args.learning_rate) if args.learning_rate is not None else float(
        cfg["learning_rate"]
    )

    if args.sweep:
        drs: list[float] = [float(x) for x in cfg.get("dropout_rates_to_test", [0.1, 0.2, 0.3])]
    else:
        drs = [float(args.dropout) if args.dropout is not None else d_out]

    if (
        args.batch_size is not None
        or args.epochs is not None
        or args.max_length is not None
    ):
        cfg = dict(cfg)
        if args.batch_size is not None:
            cfg["batch_size"] = int(args.batch_size)
        if args.epochs is not None:
            cfg["epochs"] = int(args.epochs)
        if args.max_length is not None:
            cfg["max_length"] = int(args.max_length)

    summaries: list[dict[str, Any]] = []
    for d in drs:
        print(f"\n==== Training dropout head={d}  lr={base_lr} ====")
        summ = train_one_run(
            cfg,
            dropout=d,
            learning_rate=base_lr,
            output_dir=out_dir,
            autoretry_lr=not args.no_autoretry,
            force_cpu=bool(getattr(args, "cpu", False)),
        )
        summaries.append(summ)
        log_lines = [
            f"- model: {cfg.get('model_name')}",
            f"- head dropout: {d}",
            f"- learning rate: {base_lr}",
            f"- best epoch: {summ['best_epoch']}",
            f"- best val macro F1: {summ['best_val_macro_f1']:.4f}",
            f"- device: {summ['device']}",
        ]
        for row in summ["per_epoch"]:
            log_lines.append(
                f"  - epoch {int(row['epoch'])}: loss={row['train_loss']:.4f}  "
                f"val_f1={row['val_macro_f1']:.4f}"
            )
        append_training_log(
            Path(cfg.get("log_path", "reports/training_log.md")),
            f"Train run (dropout={d})",
            log_lines,
        )

    if len(summaries) > 1:
        best = max(summaries, key=lambda s: s["best_val_macro_f1"])
        print(
            f"\n[SWEEP] Best dropout={best['dropout']:.3f}  "
            f"best val macro F1={best['best_val_macro_f1']:.4f}  (epoch {best['best_epoch']})"
        )
        append_training_log(
            Path(cfg.get("log_path", "reports/training_log.md")),
            "Dropout sweep summary",
            [
                f"Best val macro F1 across dropout_rates_to_test: {best['best_val_macro_f1']:.4f}",
                f"Best head dropout: {best['dropout']}",
            ],
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI for training."""
    p = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on 20 Newsgroup JSON (train/val)."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "training_config.yaml",
        help="Path to YAML config (default: configs/training_config.yaml).",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override classification-head dropout (default: from YAML).",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override base learning rate (default: from YAML).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (e.g. 8 on OOM).",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run consecutively for each value in dropout_rates_to_test.",
    )
    p.add_argument(
        "--no-autoretry",
        action="store_true",
        help="Do not auto-retry with half LR if val F1 is below 0.7.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (default: YAML).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (use if MPS OOM; slower).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override tokenizer max length (default: YAML).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Script entry: parse args and start training."""
    return run_from_args(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
