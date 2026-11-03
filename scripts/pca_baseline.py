
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(X_train, n_components=16, seed=42):
    """Fit PCA on training features only."""
    model = PCA(
        n_components=n_components,
        random_state=seed,
    )
    model.fit(np.asarray(X_train, dtype=np.float32))
    return model


def transform_pca(model, X):
    """Generate PCA embeddings."""
    return model.transform(
        np.asarray(X, dtype=np.float32)
    ).astype(np.float32)


def pca_summary(model):
    return {
        "dimensions": model.n_components_,
        "explained_variance": float(
            model.explained_variance_ratio_.sum()
        ),
    }
