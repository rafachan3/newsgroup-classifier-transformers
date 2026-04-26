# 20 Newsgroups Transformer Classifier

Fine-tuned transformer models for classifying 20 Newsgroup posts, with
reproducible training, evaluation, and interpretability hooks.

## Setup

- **Python**: 3.10 or newer (3.9 is not supported). On macOS you can use
  `python3.12` or `python3.10` if the default `python3` is older.
- **Virtual environment** (recommended):

  ```bash
  python3.12 -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **GPU (optional)**: A CUDA-capable GPU speeds training; the pipeline falls
  back to CPU when CUDA is not available.

Further usage (data preparation, training, evaluation) will be documented as
the project is built out in subsequent steps.

## Repository layout (planned)

- `src/` — training, evaluation, and preprocessing scripts
- `configs/` — YAML training configuration
- `data/raw/` — downloaded datasets (gitignored)
- `data/processed/` — train/validation/test JSON splits
- `models/` — saved checkpoints (gitignored)
- `reports/` — metrics, figures, and training notes
- `tests/` — automated tests
