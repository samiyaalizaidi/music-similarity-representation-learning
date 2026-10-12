# FMA Music Similarity — Starter Pipeline

## Setup

Download `fma_metadata.zip` from the [official FMA repository](https://github.com/mdeff/fma) and extract it so that these files exist:

```
project/
├── prepare_data.py
├── baseline.py
├── requirements.txt
└── data/
    └── fma_metadata/
        ├── features.csv
        ├── tracks.csv
        └── genres.csv
```

You **do not** need MP3 files for this feature-based starter experiment.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_data.py
python baseline.py --k 10
# Optionally choose a test-track ID shown in data/processed/metadata.csv:
python baseline.py --track-id 123 --k 10
```

The last command only works if track ID 123 is in the test split; substitute a real test-track ID.

## Outputs

`data/processed/training.npz`, `validation.npz`, and `test.npz` each contain `X` (standardized float32 feature matrix) and `track_ids`. `all.npz` additionally includes `split`. `metadata.csv` contains song metadata, and `preprocessing.joblib` stores the training-only feature filtering, imputation, and standardization parameters.

## Design choices

- Uses **FMA-small** for initial development, selected by FMA's `set/subset` metadata.
- Respects FMA's official training/validation/test splits.
- Fits preprocessing only on the training subset to avoid leakage.
- Baseline uses cosine similarity and retrieves neighbors from training tracks for a held-out test query.
- Genre consistency is an illustrative proxy metric, **not** a ground-truth measure of perceptual similarity.
- Tags are retained as metadata for later evaluation; verify their availability and quality before treating tag overlap as a main metric.
- PCA is deliberately postponed.
