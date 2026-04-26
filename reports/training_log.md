# Training log

Decisions, failures, and adjustments from training and experiments will be
recorded here as the pipeline is implemented.


## Phase 3 implementation notes (2026-04-26)

- Added DistilBERT training stack (`src/model.py`, `src/train.py`) and YAML
  config (`configs/training_config.yaml`) with AdamW, linear warmup, early
  stopping on validation macro F1, per-epoch checkpoints, and dropout sweeps.
- Regenerated `data/processed/{train,validation,test}.json` via
  `python src/preprocess.py` because only `.gitkeep` existed in this checkout.
- Dependency compatibility adjustment:
  - `transformers 5.x` disabled PyTorch path with `torch 2.2.2`.
  - Pinned runtime to `transformers 4.57.6` and `numpy 1.26.4` to restore
    DistilBERT training compatibility in this environment.

## Runtime blockers observed

- Device detected: no CUDA GPU (`torch.cuda.is_available() == False`), MPS
  available (`torch.backends.mps.is_available() == True`).
- MPS repeatedly ran out of memory during AdamW optimizer updates at batch
  sizes 16, 8, and 4 (including reduced max_length), preventing completion of
  full training runs.
- Forcing CPU avoided MPS OOM but throughput was too low for practical
  completion within this session window (single-epoch run did not complete in
  acceptable time).

## Actions taken for failure handling

- Applied OOM fallback by progressively reducing batch size and sequence
  length.
- Added `--cpu`, `--batch-size`, `--max-length`, and `--epochs` overrides in
  `src/train.py` to unblock execution under constrained hardware.
- Logged blocker instead of fabricating validation metrics (per project rule:
  no invented results).

## Required next step to finish Phase 3 metrics

- Run the three required dropout experiments on a CUDA-capable machine:
  - `python src/train.py --dropout 0.1 --config configs/training_config.yaml`
  - `python src/train.py --dropout 0.2 --config configs/training_config.yaml`
  - `python src/train.py --dropout 0.3 --config configs/training_config.yaml`
- Then append observed validation macro F1 values and best dropout selection.
