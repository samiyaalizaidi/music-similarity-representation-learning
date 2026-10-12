"""Prepare FMA-small's precomputed audio features for similarity learning."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def load_fma(data_dir: Path):
    # FMA's official CSVs have multi-row column headers.
    features = pd.read_csv(data_dir / 'features.csv', index_col=0, header=[0, 1, 2])
    tracks = pd.read_csv(data_dir / 'tracks.csv', index_col=0, header=[0, 1])
    features.index = pd.to_numeric(features.index, errors='raise')
    tracks.index = pd.to_numeric(tracks.index, errors='raise')
    features.index.name = tracks.index.name = 'track_id'
    return features, tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data/fma_metadata'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features, tracks = load_fma(args.data_dir)
    small = tracks.loc[tracks[('set', 'subset')].eq('small')].copy()
    joined_ids = small.index.intersection(features.index).sort_values()
    meta = small.loc[joined_ids]
    x = features.loc[joined_ids].apply(pd.to_numeric, errors='coerce')
    splits = meta[('set', 'split')].astype(str)
    valid_splits = splits.isin(['training', 'validation', 'test'])
    x, meta, splits = x.loc[valid_splits], meta.loc[valid_splits], splits.loc[valid_splits]
    print('Selected tracks:', len(x), '| split counts:', splits.value_counts().to_dict())
    print('Original feature dimensions:', x.shape[1])
    print('Missing values:', int(x.isna().sum().sum()), '| infinite values:', int(np.isinf(x.to_numpy(dtype=float)).sum()))

    # Replace inf with NaN; learn cleaning parameters ONLY on training data.
    x = x.replace([np.inf, -np.inf], np.nan)
    train = x.loc[splits.eq('training')]
    if train.empty or not splits.eq('test').any():
        raise ValueError('Expected nonempty training and test splits; check FMA metadata.')
    # Drop features that are entirely absent or constant in training.
    medians = train.median(axis=0).fillna(0.0)
    train_filled = train.fillna(medians)
    keep = train.notna().any(axis=0) & train_filled.var(axis=0).gt(1e-12)
    if not keep.any():
        raise ValueError('No usable feature columns.')
    x = x.loc[:, keep].fillna(medians.loc[keep])
    scaler = StandardScaler()
    scaler.fit(x.loc[splits.eq('training')].to_numpy(dtype=np.float64))
    x_scaled = scaler.transform(x.to_numpy(dtype=np.float64)).astype(np.float32)
    if not np.isfinite(x_scaled).all():
        raise ValueError('Non-finite values remain after preprocessing.')

    ids = x.index.to_numpy(dtype=np.int64)
    for split in ['training', 'validation', 'test']:
        mask = splits.to_numpy() == split
        np.savez_compressed(args.output_dir / f'{split}.npz', X=x_scaled[mask], track_ids=ids[mask])
        print(f'{split}: {mask.sum()} tracks')
    # One combined file for lookup; candidate retrieval can use training only.
    np.savez_compressed(args.output_dir / 'all.npz', X=x_scaled, track_ids=ids, split=splits.to_numpy())
    joblib.dump({'scaler': scaler, 'imputation_medians': medians.loc[keep], 'feature_columns': x.columns}, args.output_dir / 'preprocessing.joblib')

    # Flatten only the metadata fields we actually use; preserve track_id.
    fields = {'split': ('set', 'split'), 'genre': ('track', 'genre_top'),
              'title': ('track', 'title'), 'artist': ('artist', 'name'),
              'tags': ('track', 'tags')}
    metadata = pd.DataFrame(index=x.index)
    for name, key in fields.items():
        metadata[name] = meta[key] if key in meta.columns else ''
    metadata.to_csv(args.output_dir / 'metadata.csv', index_label='track_id')
    print('Saved to', args.output_dir, '| usable dimensions:', x.shape[1])


if __name__ == '__main__':
    main()
