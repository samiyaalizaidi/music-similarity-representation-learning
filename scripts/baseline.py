"""Retrieve similar songs using standardized original FMA features."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'processed')
    parser.add_argument('--track-id', type=int, default=None)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    data = np.load(args.processed_dir / 'all.npz')
    X, ids, splits = data['X'], data['track_ids'], data['split']
    metadata = pd.read_csv(args.processed_dir / 'metadata.csv', index_col='track_id')
    # Fixed candidate pool: training songs only; query tracks: test split only.
    candidates = splits == 'training'
    queries = splits == 'test'
    if args.track_id is None:
        query_idx = int(np.flatnonzero(queries)[0])
    else:
        found = np.flatnonzero((ids == args.track_id) & queries)
        if len(found) == 0:
            raise ValueError('track-id must belong to the test split.')
        query_idx = int(found[0])
    candidate_ids = ids[candidates]
    if args.k < 1 or args.k > len(candidate_ids):
        raise ValueError('k must be between 1 and number of training tracks')
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=args.k)
    model.fit(X[candidates])
    distances, indices = model.kneighbors(X[query_idx:query_idx + 1])
    qid = int(ids[query_idx])
    print('Query:', qid, metadata.loc[qid, ['title', 'artist', 'genre']].to_dict())
    rows = []
    for distance, idx in zip(distances[0], indices[0]):
        nid = int(candidate_ids[idx])
        row = metadata.loc[nid]
        rows.append({'track_id': nid, 'title': row['title'], 'artist': row['artist'],
                     'genre': row['genre'], 'cosine_similarity': round(float(1 - distance), 4),
                     'same_genre': bool(row['genre'] == metadata.loc[qid, 'genre'])})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    print('Genre consistency@{}: {:.3f}'.format(args.k, result['same_genre'].mean()))


if __name__ == '__main__':
    main()
