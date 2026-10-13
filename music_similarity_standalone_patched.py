#!/usr/bin/env python3
"""Learning Musical Similarity Beyond Genre -- standalone pipeline.

A single-file, shareable version of the representation-comparison pipeline.
It compares three ways of representing FMA audio features on the same
nearest-neighbor similarity task:

    1. original   -- standardized audio features (baseline)
    2. pca        -- linear dimensionality reduction
    3. autoencoder-- a learned nonlinear embedding (PyTorch)

Each is scored with metadata proxies for musical similarity: genre
consistency, tag overlap, and a cross-genre analysis.

------------------------------------------------------------------------------
QUICK START
------------------------------------------------------------------------------
Install deps (once):
    pip install numpy pandas scikit-learn torch

Run on synthetic data (no download needed, good for a smoke test):
    python music_similarity_standalone.py --synthetic

Run on real FMA metadata:
    1. Download + unzip fma_metadata.zip from https://github.com/mdeff/fma
       (direct: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip, 342 MiB)
    2. Point the script at the folder containing tracks.csv / features.csv:
         python music_similarity_standalone.py --data-dir /path/to/fma_metadata

Common options:
    --subset {small,medium,large}   FMA subset to use         (default: medium)
    --latent-dim INT                autoencoder / PCA dims    (default: 16)
    --epochs INT                    autoencoder epochs        (default: 50)
    --k INT                         neighbors per query       (default: 10)
    --out PATH                      write results CSV here     (optional)

Results print as a table and, with --out, save to CSV.
------------------------------------------------------------------------------
Dataset: Free Music Archive (FMA), metadata under CC BY 4.0.
Defferrard et al., ISMIR 2017. https://github.com/mdeff/fma
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# Heavy / optional imports are done lazily inside functions so that
# `--help` works even if torch isn't installed yet.


# ===========================================================================
# Data loading
# ===========================================================================
# FMA subsets are nested: small ⊂ medium ⊂ large. tracks.csv records the
# smallest subset each track belongs to, so "medium" means {small, medium}.
_SUBSET_ORDER = {"small": 0, "medium": 1, "large": 2}


@dataclass
class TrackData:
    """Aligned per-track features and labels."""

    track_ids: np.ndarray
    features: np.ndarray
    feature_names: list
    genres: np.ndarray
    tags: list

    def __len__(self) -> int:
        return len(self.track_ids)


def _find_metadata_dir(metadata_dir: Path) -> Path:
    """Locate the directory that actually contains tracks.csv (may be nested)."""
    if (metadata_dir / "tracks.csv").exists():
        return metadata_dir
    matches = list(metadata_dir.rglob("tracks.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find tracks.csv under {metadata_dir}. "
            "Download and unzip fma_metadata.zip first."
        )
    return matches[0].parent


def _parse_tags(raw: Any) -> list:
    """FMA tag cells are stringified Python lists, e.g. "['rock', 'live']"."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(t) for t in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def _flatten_columns(df) -> list:
    names = []
    for col in df.columns:
        if isinstance(col, tuple):
            names.append("_".join(str(c) for c in col))
        else:
            names.append(str(col))
    return names


def load_real(metadata_dir: Path, subset: str = "medium",
              feature_source: str = "features") -> TrackData:
    """Load and align real FMA tables, restricted to the requested subset."""
    import pandas as pd

    metadata_dir = _find_metadata_dir(metadata_dir)
    tracks = pd.read_csv(metadata_dir / "tracks.csv", index_col=0, header=[0, 1])

    for col in [("track", "genre_top"), ("set", "subset"), ("set", "split")]:
        if col in tracks.columns:
            tracks[col] = tracks[col].astype("category")

    # Restrict to the requested subset (and all smaller nested subsets).
    if ("set", "subset") in tracks.columns and subset in _SUBSET_ORDER:
        max_rank = _SUBSET_ORDER[subset]
        s = tracks[("set", "subset")].astype(str)
        tracks = tracks[s.map(lambda v: _SUBSET_ORDER.get(v, 99) <= max_rank)]

    if feature_source == "echonest":
        feats = pd.read_csv(metadata_dir / "echonest.csv", index_col=0, header=[0, 1, 2])
        if ("echonest", "audio_features") in feats.columns:
            feats = feats["echonest"]["audio_features"]
    else:
        feats = pd.read_csv(metadata_dir / "features.csv", index_col=0, header=[0, 1, 2])

    common = tracks.index.intersection(feats.index)
    tracks = tracks.loc[common]
    feats = feats.loc[common]

    if ("track", "genre_top") in tracks.columns:
        genres = tracks[("track", "genre_top")].astype(str).fillna("").to_numpy()
    else:
        genres = np.array([""] * len(tracks))

    if ("track", "tags") in tracks.columns:
        tags = [_parse_tags(v) for v in tracks[("track", "tags")].tolist()]
    else:
        tags = [[] for _ in range(len(tracks))]

    return TrackData(
        track_ids=np.asarray(common, dtype=np.int64),
        features=feats.to_numpy(dtype=np.float64),
        feature_names=_flatten_columns(feats),
        genres=np.asarray(genres),
        tags=tags,
    )


def make_synthetic(n_tracks: int = 2000, n_features: int = 100,
                   n_genres: int = 8, seed: int = 42) -> TrackData:
    """Synthetic track data with genre-clustered structure (for smoke tests)."""
    rng = np.random.default_rng(seed)
    genre_names = [f"genre_{i}" for i in range(n_genres)]
    # Overlapping genre structure: modest center separation, large per-track
    # noise, and a shared cross-genre latent factor so neighbors are NOT
    # trivially genre-pure. This keeps genre@k < 1 and produces real
    # cross-genre neighbors, so the comparison can actually discriminate
    # between the representations.
    centers = rng.normal(0, 1.5, size=(n_genres, n_features))
    genre_idx = rng.integers(0, n_genres, size=n_tracks)
    # A few latent directions shared across ALL genres (blurs boundaries).
    n_shared = 5
    shared_dirs = rng.normal(0, 1.0, size=(n_shared, n_features))
    shared_coeffs = rng.normal(0, 2.0, size=(n_tracks, n_shared))
    features = (
        centers[genre_idx]
        + shared_coeffs @ shared_dirs
        + rng.normal(0, 3.0, size=(n_tracks, n_features))
    )

    cross_cutting = ["live", "instrumental", "lo-fi", "remix"]
    genre_tags = {g: [f"{genre_names[g]}_tag_{j}" for j in range(3)] for g in range(n_genres)}
    tags = []
    for g in genre_idx:
        chosen = set()
        for t in genre_tags[g]:
            if rng.random() < 0.6:
                chosen.add(t)
        for t in cross_cutting:
            if rng.random() < 0.15:
                chosen.add(t)
        tags.append(sorted(chosen))

    return TrackData(
        track_ids=np.arange(n_tracks, dtype=np.int64),
        features=features,
        feature_names=[f"feat_{i}" for i in range(n_features)],
        genres=np.asarray([genre_names[g] for g in genre_idx]),
        tags=tags,
    )


# ===========================================================================
# Preprocessing
# ===========================================================================
@dataclass
class Dataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    genres_train: np.ndarray
    genres_val: np.ndarray
    genres_test: np.ndarray
    tags_train: list
    tags_val: list
    tags_test: list

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]


def _fit_impute_means(features: np.ndarray) -> np.ndarray:
    """Compute per-column imputation means from the given rows (train only)."""
    f = features.astype(np.float64, copy=True)
    f[~np.isfinite(f)] = np.nan
    col_means = np.nanmean(f, axis=0)
    return np.where(np.isfinite(col_means), col_means, 0.0)


def _apply_impute(features: np.ndarray, col_means: np.ndarray) -> np.ndarray:
    """Replace inf/NaN with the provided (train) column means."""
    f = features.astype(np.float64, copy=True)
    f[~np.isfinite(f)] = np.nan
    inds = np.where(np.isnan(f))
    f[inds] = np.take(col_means, inds[1])
    return f


def preprocess(data: TrackData, val_size: float = 0.1, test_size: float = 0.1,
               random_state: int = 42) -> Dataset:
    """Clean, (genre-stratified) split, and standardize using train statistics."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_raw = data.features.astype(np.float64, copy=True)
    genres = data.genres
    all_idx = np.arange(len(X_raw))

    _, counts = np.unique(genres, return_counts=True)
    stratify = genres if counts.min() >= 2 else None

    tv_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state, stratify=stratify)

    strat_tv = genres[tv_idx] if stratify is not None else None
    if strat_tv is not None:
        _, c = np.unique(strat_tv, return_counts=True)
        if c.min() < 2:
            strat_tv = None
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        tv_idx, test_size=rel_val, random_state=random_state, stratify=strat_tv)

    # Impute using TRAIN column means only (no leakage), then standardize
    # (also train-only). Both statistics are fit on the training rows and
    # applied to val/test.
    impute_means = _fit_impute_means(X_raw[train_idx])
    X_train = _apply_impute(X_raw[train_idx], impute_means)
    X_val = _apply_impute(X_raw[val_idx], impute_means)
    X_test = _apply_impute(X_raw[test_idx], impute_means)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pick = lambda items, idx: [items[i] for i in idx]
    return Dataset(
        X_train=X_train, X_val=X_val, X_test=X_test,
        genres_train=genres[train_idx], genres_val=genres[val_idx], genres_test=genres[test_idx],
        tags_train=pick(data.tags, train_idx), tags_val=pick(data.tags, val_idx),
        tags_test=pick(data.tags, test_idx),
    )


# ===========================================================================
# Representations
# ===========================================================================
class OriginalFeatures:
    name = "original"

    def fit(self, X_train):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)


class PCARepresentation:
    name = "pca"

    def __init__(self, n_components: int = 16, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._pca = None

    def fit(self, X_train):
        from sklearn.decomposition import PCA
        n = min(self.n_components, min(X_train.shape))
        self._pca = PCA(n_components=n, random_state=self.random_state)
        self._pca.fit(X_train)
        return self

    def transform(self, X):
        return self._pca.transform(X)

    @property
    def explained_variance_ratio(self):
        return self._pca.explained_variance_ratio_


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 16
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5          # AdamW decoupled weight decay
    patience: int = 8                   # early-stopping patience (epochs)
    lr_patience: int = 5                # ReduceLROnPlateau patience (epochs)
    lr_factor: float = 0.5              # LR multiplier on plateau
    device: str = "cpu"
    seed: int = 42


class AutoencoderRepresentation:
    """Symmetric fully-connected autoencoder; the bottleneck is the embedding."""

    name = "autoencoder"

    def __init__(self, cfg: AutoencoderConfig):
        import torch
        import torch.nn as nn

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.history = {"train_loss": [], "val_loss": []}

        def mlp(dims, final_activation):
            layers = []
            for i in range(len(dims) - 1):
                is_last = i == len(dims) - 2
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if not is_last or final_activation:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                    layers.append(nn.ReLU(inplace=True))
                    if cfg.dropout > 0:
                        layers.append(nn.Dropout(cfg.dropout))
            return nn.Sequential(*layers)

        enc_dims = [cfg.input_dim, *cfg.hidden_dims, cfg.latent_dim]
        dec_dims = [cfg.latent_dim, *reversed(cfg.hidden_dims), cfg.input_dim]

        class _AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = mlp(enc_dims, final_activation=False)
                self.decoder = mlp(dec_dims, final_activation=False)

            def encode(self, x):
                return self.encoder(x)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        self.model = _AE().to(self.device)

    def fit(self, X_train, X_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.cfg
        loader = DataLoader(
            TensorDataset(torch.as_tensor(X_train, dtype=torch.float32)),
            batch_size=cfg.batch_size, shuffle=True,
            drop_last=X_train.shape[0] > cfg.batch_size)  # BatchNorm needs >1 sample
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
        # Decay the LR when validation loss plateaus. Kept separate from (and
        # complementary to) early stopping: the scheduler reshapes the descent,
        # early stopping decides when to quit and which weights to keep.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        stale = 0
        for _ in range(cfg.epochs):
            self.model.train()
            running, seen = 0.0, 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                loss = criterion(self.model(batch), batch)
                loss.backward()
                opt.step()
                running += loss.item() * batch.size(0)
                seen += batch.size(0)
            self.history["train_loss"].append(running / max(seen, 1))

            if X_val is not None and len(X_val) > 0:
                val = self._eval_loss(X_val)
                self.history["val_loss"].append(val)
                scheduler.step(val)
                if val < best_val - 1e-6:
                    best_val = val
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if cfg.patience > 0 and stale >= cfg.patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _eval_loss(self, X):
        import torch
        import torch.nn as nn
        self.model.eval()
        with torch.no_grad():
            t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return nn.MSELoss()(self.model(t), t).item()

    def transform(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self.model.encode(t).cpu().numpy()

    def reconstruction_error(self, X):
        return self._eval_loss(X)


def build_autoencoder(input_dim: int, latent_dim: int, hidden_dims: list,
                      epochs: int, seed: int) -> AutoencoderRepresentation:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AutoencoderRepresentation(AutoencoderConfig(
        input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
        epochs=epochs, device=device, seed=seed))


# ===========================================================================
# Similarity + evaluation
# ===========================================================================
class NeighborIndex:
    """Nearest-neighbor retrieval; cosine inputs are L2-normalized for stability."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self._nn = None
        self._reference = None

    def _prepare(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if self.metric == "cosine":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        return X

    def fit(self, embeddings):
        from sklearn.neighbors import NearestNeighbors
        self._reference = self._prepare(embeddings)
        self._nn = NearestNeighbors(metric=self.metric)
        self._nn.fit(self._reference)
        return self

    def query(self, queries, k, exclude_self=True, query_ids=None):
        # query_ids: absolute indices of `queries` within the fitted reference set,
        # used to exclude the exact self-match by identity. Defaults to the first
        # len(queries) reference rows (the common "query == reference" case).
        if query_ids is None:
            query_ids = np.arange(queries.shape[0] if hasattr(queries, "shape") else len(queries))
        self._query_ids = np.asarray(query_ids)
        queries = self._prepare(queries)
        k_search = min((k + 1 if exclude_self else k), self._reference.shape[0])
        dists, idxs = self._nn.kneighbors(queries, n_neighbors=k_search)
        if not np.isfinite(dists).all():
            raise FloatingPointError(
                "Non-finite neighbor distances encountered. This usually means a "
                "degenerate embedding (e.g. an autoencoder collapse). Investigate "
                "the representation rather than masking the distances."
            )

        if not exclude_self:
            return idxs[:, :k], dists[:, :k]

        # FIX 2: exclude the self-match by IDENTITY, not by a distance threshold.
        # A zero/degenerate vector has cosine self-distance 1.0 (not ~0), so the
        # old `dists[i][0] <= 1e-9` test would silently keep the self-match and
        # drop a real neighbor. We instead pass in the absolute row indices of the
        # queries and remove exactly those.
        out_idx = np.empty((queries.shape[0], k), dtype=np.int64)
        self_ids = self._query_ids
        for i in range(queries.shape[0]):
            row = [j for j in idxs[i] if j != self_ids[i]]
            # If the self id wasn't among the retrieved neighbors (query not in the
            # reference set), just drop the farthest to keep exactly k.
            if len(row) > k:
                row = row[:k]
            out_idx[i] = row[:k]
        return out_idx, None


@dataclass
class EvalResult:
    representation: str
    embedding_dim: int
    k: int
    n_queries: int
    genre_consistency_at_k: float
    tag_overlap_at_k: float
    cross_genre_fraction: float
    cross_genre_tag_overlap: float
    cross_genre_with_shared_tags: float

    def as_row(self):
        return asdict(self)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def evaluate_representation(embeddings, genres, tags, name, k=10,
                            metric="cosine", n_queries=0, random_state=42) -> EvalResult:
    n = len(embeddings)
    index = NeighborIndex(metric=metric).fit(embeddings)
    rng = np.random.default_rng(random_state)
    rows = (rng.choice(n, size=n_queries, replace=False)
            if n_queries and n_queries < n else np.arange(n))
    neighbor_idx, _ = index.query(embeddings[rows], k=k, exclude_self=True,
                                  query_ids=rows)
    tag_sets = [set(t) for t in tags]

    genre_hits = genre_total = 0
    tag_sum = tag_count = 0.0
    xg_total = 0
    xg_tag_sum = 0.0
    xg_shared = 0
    for qi, q in enumerate(rows):
        q_genre, q_tags = genres[q], tag_sets[q]
        for nb in neighbor_idx[qi]:
            genre_total += 1
            same = genres[nb] == q_genre
            if same:
                genre_hits += 1
            ov = _jaccard(q_tags, tag_sets[nb])
            tag_sum += ov
            tag_count += 1
            if not same:
                xg_total += 1
                xg_tag_sum += ov
                if len(q_tags & tag_sets[nb]) > 0:
                    xg_shared += 1

    g = (lambda num, den: num / den if den else 0.0)
    return EvalResult(
        representation=name, embedding_dim=int(embeddings.shape[1]), k=k,
        n_queries=int(len(rows)),
        genre_consistency_at_k=round(g(genre_hits, genre_total), 4),
        tag_overlap_at_k=round(g(tag_sum, tag_count), 4),
        cross_genre_fraction=round(g(xg_total, genre_total), 4),
        cross_genre_tag_overlap=round(g(xg_tag_sum, xg_total), 4),
        cross_genre_with_shared_tags=round(g(xg_shared, xg_total), 4),
    )


def format_comparison(results) -> str:
    headers = [
        ("representation", "representation", 16),
        ("embedding_dim", "dim", 6),
        ("genre_consistency_at_k", "genre@k", 9),
        ("tag_overlap_at_k", "tagJac@k", 9),
        ("cross_genre_fraction", "xgenre%", 9),
        ("cross_genre_tag_overlap", "xg_tagJac", 10),
        ("cross_genre_with_shared_tags", "xg_shared", 10),
    ]
    header_line = "  ".join(f"{title:>{w}}" for _, title, w in headers)
    lines = [header_line, "-" * len(header_line)]
    for r in results:
        row = r.as_row()
        cells = []
        for key, _, w in headers:
            v = row[key]
            cells.append(f"{v:>{w}.4f}" if isinstance(v, float) else f"{str(v):>{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare original / PCA / autoencoder representations for FMA music similarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", type=str,
                     help="folder containing FMA tracks.csv/features.csv (the unzipped fma_metadata)")
    src.add_argument("--synthetic", action="store_true",
                     help="use generated synthetic data instead of real FMA (no download needed)")

    p.add_argument("--subset", choices=["small", "medium", "large"], default="medium")
    p.add_argument("--feature-source", choices=["features", "echonest"], default="features")
    p.add_argument("--latent-dim", type=int, default=16,
                   help="embedding size for both PCA and the autoencoder")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64],
                   help="autoencoder hidden layer sizes")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--k", type=int, default=10, help="neighbors retrieved per query")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--n-queries", type=int, default=500,
                   help="query tracks sampled for evaluation (0 = all test tracks)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="optional path to write results CSV")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)

    print("Loading data...")
    if args.synthetic:
        data = make_synthetic(seed=args.seed)
        source = "synthetic"
    else:
        data = load_real(Path(args.data_dir), subset=args.subset,
                         feature_source=args.feature_source)
        source = args.feature_source
    print(f"  source={source}  tracks={len(data)}  features={data.features.shape[1]}")

    dataset = preprocess(data, random_state=args.seed)
    print(f"  split: train={len(dataset.X_train)} val={len(dataset.X_val)} "
          f"test={len(dataset.X_test)}")

    print("Building representations...")
    embeddings = {}

    embeddings["original"] = OriginalFeatures().fit(dataset.X_train).transform(dataset.X_test)

    pca = PCARepresentation(n_components=args.latent_dim, random_state=args.seed).fit(dataset.X_train)
    embeddings["pca"] = pca.transform(dataset.X_test)
    print(f"  PCA: {pca._pca.n_components_} comps, "
          f"{float(np.sum(pca.explained_variance_ratio)):.1%} variance retained")

    ae = build_autoencoder(dataset.input_dim, args.latent_dim, args.hidden_dims,
                           args.epochs, args.seed)
    print(f"  Training autoencoder (device={ae.cfg.device})...")
    ae.fit(dataset.X_train, dataset.X_val)
    if len(dataset.X_test):
        print(f"  Autoencoder: test reconstruction MSE={ae.reconstruction_error(dataset.X_test):.4f}")
    embeddings["autoencoder"] = ae.transform(dataset.X_test)

    print("Evaluating nearest-neighbor similarity on the test split...")
    results = [
        evaluate_representation(emb, dataset.genres_test, dataset.tags_test, name,
                                k=args.k, metric=args.metric, n_queries=args.n_queries,
                                random_state=args.seed)
        for name, emb in embeddings.items()
    ]

    print("\n=== Representation comparison (test split) ===")
    print(format_comparison(results))
    print("\nReading the table: higher genre@k means neighbors mostly share genre; "
          "higher tagJac@k / xg_shared means similarity captured beyond genre.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.as_row() for r in results]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
