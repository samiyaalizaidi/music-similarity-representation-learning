
import ast
import re

import numpy as np
import pandas as pd


def parse_tags(value):
    """Convert metadata tags into a normalized set."""
    if value is None:
        return set()

    if isinstance(value, (list, tuple, set)):
        values = value
    elif isinstance(value, str):
        value = value.strip()

        if not value or value.lower() in {"nan", "none", "[]"}:
            return set()

        if value.startswith("["):
            try:
                values = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                values = [value]
        else:
            values = re.split(r"[,;|]", value)
    else:
        if pd.isna(value):
            return set()
        values = [value]

    if isinstance(values, str):
        values = [values]

    return {
        str(tag).strip().lower()
        for tag in values
        if str(tag).strip()
    }


def jaccard_similarity(tags_a, tags_b):
    """Return NaN when either track has no usable tags."""
    a = parse_tags(tags_a)
    b = parse_tags(tags_b)

    if not a or not b:
        return np.nan

    return len(a & b) / len(a | b)


def evaluate_retrieval(
    retrieval_df,
    metadata,
    k=10,
    genre_column="genre",
    tags_column="tags",
):
    """
    Evaluate genre consistency, cross-genre retrieval,
    and tag overlap.

    Metadata must be indexed by track ID.
    """
    if not metadata.index.is_unique:
        raise ValueError("Metadata track IDs must be unique.")

    if genre_column not in metadata.columns:
        raise KeyError(f"Missing metadata column: {genre_column}")

    df = retrieval_df[
        retrieval_df["rank"] <= k
    ].copy()

    if df.empty:
        raise ValueError("No retrieval results to evaluate.")

    genres = metadata[genre_column]

    df["query_genre"] = df["query_id"].map(genres)
    df["neighbor_genre"] = df["neighbor_id"].map(genres)

    valid_genre = (
        df["query_genre"].notna()
        & df["neighbor_genre"].notna()
    )

    df["same_genre"] = np.nan
    df.loc[valid_genre, "same_genre"] = (
        df.loc[valid_genre, "query_genre"]
        == df.loc[valid_genre, "neighbor_genre"]
    ).astype(float)

    df["cross_genre"] = 1.0 - df["same_genre"]

    if tags_column in metadata.columns:
        tags = metadata[tags_column]

        df["query_tags"] = df["query_id"].map(tags)
        df["neighbor_tags"] = df["neighbor_id"].map(tags)

        df["tag_overlap"] = [
            jaccard_similarity(a, b)
            for a, b in zip(
                df["query_tags"], df["neighbor_tags"]
            )
        ]
    else:
        df["tag_overlap"] = np.nan

    cross_mask = df["cross_genre"] == 1.0

    metrics = {
        "k": k,
        "n_queries": int(df["query_id"].nunique()),
        "n_pairs": len(df),
        "genre_consistency": df["same_genre"].mean(),
        "cross_genre_rate": df["cross_genre"].mean(),
        "tag_overlap": df["tag_overlap"].mean(),
        "cross_genre_tag_overlap": (
            df.loc[cross_mask, "tag_overlap"].mean()
        ),
        "tag_coverage": df["tag_overlap"].notna().mean(),
        "cross_genre_tag_coverage": (
            df.loc[cross_mask, "tag_overlap"].notna().mean()
        ),
    }

    return metrics, df
