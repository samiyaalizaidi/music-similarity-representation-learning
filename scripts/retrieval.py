
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def retrieve_neighbors(
    query_embeddings,
    candidate_embeddings,
    query_ids,
    candidate_ids,
    k=10,
):
    """
    Retrieve nearest neighbors using cosine similarity.

    Query and candidate embeddings must come from
    the same representation, but can have any dimension.
    """
    Q = np.asarray(query_embeddings, dtype=np.float32)
    C = np.asarray(candidate_embeddings, dtype=np.float32)

    query_ids = np.asarray(query_ids, dtype=np.int64)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

    if Q.ndim != 2 or C.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")

    if Q.shape[1] != C.shape[1]:
        raise ValueError("Embedding dimensions do not match.")

    if len(Q) != len(query_ids) or len(C) != len(candidate_ids):
        raise ValueError("Track ID and embedding counts differ.")

    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate track IDs must be unique.")

    if not np.isfinite(Q).all() or not np.isfinite(C).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    if len(C) == 0 or k < 1:
        raise ValueError("Need candidates and positive k.")

    # Retrieve additional candidates so we can remove
    # a query's own track if it is in the candidate set.
    candidate_id_set = set(candidate_ids.tolist())
    overlap = any(int(q) in candidate_id_set for q in query_ids)
    n_neighbors = min(len(C), k + 1 if overlap else k)

    index = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )
    index.fit(C)

    distances, indices = index.kneighbors(Q)

    records = []

    for row, query_id in enumerate(query_ids):
        rank = 0

        for distance, candidate_index in zip(
            distances[row], indices[row]
        ):
            neighbor_id = int(candidate_ids[candidate_index])

            if neighbor_id == int(query_id):
                continue

            rank += 1

            records.append({
                "query_id": int(query_id),
                "rank": rank,
                "neighbor_id": neighbor_id,
                "cosine_similarity": float(1.0 - distance),
            })

            if rank >= k:
                break

    return pd.DataFrame(
        records,
        columns=[
            "query_id",
            "rank",
            "neighbor_id",
            "cosine_similarity",
        ],
    )
