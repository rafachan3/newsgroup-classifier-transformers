# 20 Newsgroups Transformer Classifier

A portfolio-grade NLP text classifier for the 20 Newsgroups corpus, focused on
clean engineering and reproducible experimentation. The pipeline performs
preprocessing (stopword removal + Porter stemming), stratified train/validation/
test splits, DistilBERT fine-tuning with configurable dropout, and structured
evaluation artifacts. The project also includes explainability (SHAP with LIME
fallback) and ONNX export checks to demonstrate deployment readiness. Design
choices are documented in `reports/training_log.md`, including failed runs and
hardware constraints, to keep the workflow transparent and audit-friendly.

## Setup and installation

1. Use Python 3.10+ (3.12 recommended in this repo):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Verify optional accelerators:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## Reproduce training

1. Build processed splits (writes `data/processed/*.json`):

```bash
python src/preprocess.py
```

2. Train with config defaults:

```bash
python src/train.py --config configs/training_config.yaml
```

3. Sweep dropout values from config (`[0.1, 0.2, 0.3]`):

```bash
python src/train.py --config configs/training_config.yaml --sweep
```

4. Evaluate best checkpoint on test split:

```bash
python src/evaluate.py --config configs/training_config.yaml
```

5. Run explainability + ONNX export:

```bash
python src/explain.py --config configs/training_config.yaml
```

## Results summary

- Best model family: `distilbert-base-uncased` with custom dropout+linear head.
- Validation/test metric targets are implemented in code and logs, but current
  workspace artifacts show **blocked evaluation** because no checkpoint exists
  under `models/`.
- Key findings to date:
  - Data preprocessing and stratification are reproducible and test-covered.
  - Runtime constraints (no CUDA; repeated MPS OOM during optimizer updates)
    are explicitly logged rather than hidden.
  - Evaluation/explainability scripts fail safely by writing blocker reports
    instead of fabricating metrics.

## Limitations

This repository currently lacks a completed end-to-end training run artifact in
`models/`, so final F1 numbers and confusion/attention plots are blocked in this
workspace snapshot. The baseline pipeline uses classical token normalization
(stopword removal + stemming), which may discard useful nuance for transformer
models. Hardware constraints (no CUDA in this environment) significantly limit
throughput and experiment depth.

## File structure

```text
project-root/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── preprocess.py
│   ├── augment.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── configs/
│   └── training_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── metrics/
│   ├── figures/
│   └── training_log.md
├── notebooks/
└── tests/
```
