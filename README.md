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

## Milestone 2: Results and Findings

Milestone 2 compares three music representations: original FMA features (518D), PCA (16D), and autoencoder embeddings (16D). All models use the same 800 test queries, 6,400 training candidates, and cosine similarity for top-10 retrieval.

### Retrieval Results

| Representation | Genre Consistency@10 | Cross-Genre Retrieval@10 |
|---|---:|---:|
| Original Features | 31.72% | 68.28% |
| PCA | 28.74% | 71.26% |
| Autoencoder | 26.26% | 73.74% |

The autoencoder achieved the highest cross-genre retrieval rate (73.74%), outperforming the original features by 5.46 percentage points. PCA retained 50.92% of feature variance, while the autoencoder achieved a test reconstruction MSE of 0.6370.

### Tag Similarity Evaluation

| Metric | Original | PCA | Autoencoder |
|---|---:|---:|---:|
| Tag Overlap | 0.0473 | 0.0529 | 0.0467 |
| Cross-Genre Tag Overlap | 0.0088 | 0.0066 | 0.0068 |
| Tag Coverage | 3.69% | 3.52% | 3.84% |

### Key Findings

The autoencoder retrieves more cross-genre tracks, suggesting that learned embeddings organize musical relationships differently from the original features. However, it does not improve tag overlap, and sparse metadata limits evaluation.

**Conclusion:** Learned representations change similarity retrieval patterns, but further analysis is needed to determine whether cross-genre matches reflect meaningful musical similarity.

### Audio-Based Similarity Evaluation

To further evaluate retrieval quality, we processed FMA-small MP3 recordings using `librosa` and extracted audio descriptors, including tempo, energy, spectral characteristics, chroma, and MFCCs. We evaluated 100 test queries and compared the three representations against a random cross-genre baseline.

| Metric (Lower is Better) | Original | PCA | Autoencoder | Random |
|---|---:|---:|---:|---:|
| Audio-Descriptor Distance | **0.7682** | 0.7821 | 0.8006 | 1.0517 |
| Tempo Difference (BPM) | 22.55 | 22.96 | **22.35** | 23.93 |
| RMS Energy Difference | **0.0787** | 0.0876 | 0.0866 | 0.1099 |
| Spectral Centroid Difference (Hz) | 517.46 | **485.94** | 514.00 | 767.97 |

**Key Findings:**
- All three representations retrieved tracks with closer audio characteristics than the random baseline.
- Original features achieved the best overall audio-descriptor similarity, while PCA performed best on spectral centroid similarity.
- The autoencoder achieved the lowest tempo difference but did not outperform the original features overall.
- Although the autoencoder produced the highest cross-genre retrieval rate (73.74%) in earlier experiments, this did not translate into better overall audio similarity.

**Conclusion:** The results suggest that dimensionality reduction changes retrieval behavior but may discard useful audio information. Further evaluation will focus on cross-genre-only comparisons and qualitative listening analysis.

*Note: These are preliminary results from 100 queries. The random baseline contains only cross-genre pairs, whereas model retrieval includes both same-genre and cross-genre pairs.*