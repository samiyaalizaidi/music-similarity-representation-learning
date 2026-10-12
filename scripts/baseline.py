"""Cosine-similarity baseline using standardized FMA features.

Candidates are TRAINING tracks; queries are TEST tracks. No test leakage.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[1] / 'data' / 'processed'


def load_data(processed_dir=DEFAULT_PROCESSED_DIR):
    processed_dir = Path(processed_dir)
    with np.load(processed_dir / 'all.npz') as data:
        X = data['X'].astype(np.float32)
        ids = data['track_ids'].astype(np.int64)
        splits = data['split'].astype(str)
    metadata = pd.read_csv(processed_dir / 'metadata.csv', index_col='track_id')
    if len(X) != len(ids) or len(ids) != len(splits):
        raise ValueError('Inconsistent array lengths in all.npz')
    if not np.isfinite(X).all():
        raise ValueError('Non-finite features detected')
    return X, ids, splits, metadata


def build_index(X, splits, k=10):
    candidates = np.flatnonzero(splits == 'training')
    if k < 1 or k > len(candidates):
        raise ValueError(f'k must be between 1 and {len(candidates)}')
    model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    model.fit(X[candidates])
    return model, candidates


def retrieve(X, ids, splits, metadata, model, candidate_indices, query_index, k=5):
    if splits[query_index] != 'test':
        raise ValueError('Queries must belong to the test split')
    if k < 1 or k > len(candidate_indices):
        raise ValueError('Invalid k')
    distances, positions = model.kneighbors(X[query_index:query_index + 1], n_neighbors=k)
    qid = int(ids[query_index])
    qgenre = metadata.loc[qid, 'genre']
    rows = []
    for rank, (distance, pos) in enumerate(zip(distances[0], positions[0]), 1):
        tid = int(ids[candidate_indices[pos]])
        m = metadata.loc[tid]
        rows.append({'query_track_id': qid, 'rank': rank, 'track_id': tid,
                     'title': m['title'], 'artist': m['artist'], 'genre': m['genre'],
                     'cosine_similarity': float(1 - distance),
                     'same_genre': bool(pd.notna(qgenre) and pd.notna(m['genre']) and m['genre'] == qgenre)})
    return pd.DataFrame(rows)


def evaluate_genre_consistency(X, ids, splits, metadata, model, candidate_indices, k=5):
    """Mean fraction of top-k training neighbors sharing each test query's genre.

    Excludes test queries without a known genre; this is a proxy, not ground-truth similarity.
    """
    test_indices = np.flatnonzero(splits == 'test')
    valid = [i for i in test_indices if int(ids[i]) in metadata.index and pd.notna(metadata.loc[int(ids[i]), 'genre'])]
    if not valid:
        raise ValueError('No labeled test queries')
    distances, positions = model.kneighbors(X[valid], n_neighbors=k)
    candidate_ids = ids[candidate_indices]
    candidate_genres = metadata.reindex(candidate_ids)['genre'].to_numpy()
    query_genres = metadata.reindex(ids[valid])['genre'].to_numpy()
    same = (candidate_genres[positions] == query_genres[:, None]) & pd.notna(candidate_genres[positions])
    per_query = same.mean(axis=1)
    return {'k': k, 'genre_consistency': float(per_query.mean()),
            'cross_genre_rate': float(1 - per_query.mean()), 'n_queries': len(valid)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--track-id', type=int, default=None)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    X, ids, splits, metadata = load_data(args.processed_dir)
    model, candidates = build_index(X, splits, args.k)
    test_indices = np.flatnonzero(splits == 'test')
    if not len(test_indices):
        raise ValueError('No test tracks')
    if args.track_id is None:
        query_idx = int(test_indices[0])
    else:
        matching = np.flatnonzero((ids == args.track_id) & (splits == 'test'))
        if not len(matching):
            raise ValueError('--track-id must identify a test track')
        query_idx = int(matching[0])
    qid = int(ids[query_idx])
    print('Query:', qid, metadata.loc[qid, ['title', 'artist', 'genre']].to_dict())
    result = retrieve(X, ids, splits, metadata, model, candidates, query_idx, args.k)
    print(result.drop(columns='query_track_id').to_string(index=False))
    print('Genre consistency@{}: {:.3f}'.format(args.k, result['same_genre'].mean()))


if __name__ == '__main__':
    main()
