import logging
from typing import TypedDict

import numpy as np
import umap
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class EmbeddingsResult(TypedDict):
    embeddings_df_dict: dict[str, np.ndarray]
    dims_for_embedding_dict: dict[str, int]


def _shift_to_non_negative(values: np.ndarray) -> np.ndarray:
    """Translate an array so all entries are non-negative."""
    if (values < 0).any():
        return values - values.min()
    return values



def get_dims_for_embedding(x: np.ndarray) -> dict[str, int]:
    """Compute target dimensions for a set of reduced embeddings."""

    dims_for_embedding_dict: dict[str, int] = {
        "half": int(x.shape[1] / 2),
        "sqrt": int(np.sqrt(x.shape[1])),
        "log": int(np.log(x.shape[1])),
        "ten_percent": int(x.shape[1] * 0.1),
    }
    # Correcting for spectral initialization in case the number of dimensions
    # of the embedding is not less than x.shape[0]
    for k, v in dims_for_embedding_dict.items():
        if v >= x.shape[0]:
            dims_for_embedding_dict[k] = x.shape[0] - 2
    return dims_for_embedding_dict



def reduce_dims(
    x: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
) -> np.ndarray:
    """Reduce dimensionality with UMAP."""

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "n_jobs value 1 overridden to 1 by setting random_state\. "
                "Use no seed for parallelism\."
            ),
            category=UserWarning,
        )
        u = fit.fit_transform(x)
    u = _shift_to_non_negative(u)
    logger.info("Shape of the embedding: %s", u.shape)
    return u



def reduce_dimensions_with_pca(
    X: np.ndarray,
    d: int,
    random_state: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Reduce dimensionality with PCA after standardization."""

    if d > X.shape[1]:
        raise ValueError(f"Cannot reduce to {d} dimensions when input has only {X.shape[1]} features")

    # Step 1: Standardize the data (z-score normalization)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Step 2: Apply PCA
    pca = PCA(n_components=d, random_state=random_state)
    X_reduced = pca.fit_transform(X_std)

    X_reduced = _shift_to_non_negative(X_reduced)

    if verbose:
        # Print standardization info
        logger.info(
            "Standardization complete (mean=%s, std=%s)",
            scaler.mean_.round(2),
            scaler.scale_.round(2),
        )

        # Print explained variance information
        explained_variance = pca.explained_variance_ratio_
        logger.info(
            "Explained variance ratio by each component: %s",
            explained_variance.round(4),
        )
        logger.info("Total explained variance: %.4f", explained_variance.sum())

    # Print shape information
    logger.info("Shape transformed from %s to %s", X.shape, X_reduced.shape)

    return X_reduced



def create_embeddings(
    x: np.ndarray,
    seed: int = 42,
    pca: bool = False,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = 'euclidean',
) -> EmbeddingsResult:
    """Create reduced embeddings and include the original input space."""

    logger.info("Original shape: %s", x.shape)
    dims_for_embedding_dict = get_dims_for_embedding(x)
    embeddings_df_dict: dict[str, np.ndarray] = {}

    if pca:
        logger.info("Using PCA for dimensionality reduction")
        for k, v in dims_for_embedding_dict.items():
            logger.info("Creating embedding with %s dimensions (%s)", v, k)
            embeddings_df_dict[k] = reduce_dimensions_with_pca(
                x,
                d=v,
                random_state=seed,
                verbose=False,
            )
    else:
        logger.info("Using UMAP for dimensionality reduction")
        logger.info(
            "UMAP parameters: n_neighbors=%s, min_dist=%s, metric=%s",
            umap_n_neighbors,
            umap_min_dist,
            umap_metric,
        )
        for k, v in dims_for_embedding_dict.items():
            logger.info("Creating embedding with %s dimensions (%s)", v, k)
            embeddings_df_dict[k] = reduce_dims(
                x,
                n_components=v,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                seed=seed,
            )

    dims_for_embedding_dict["original"] = x.shape[1]
    embeddings_df_dict["original"] = x
    return {
        "embeddings_df_dict": embeddings_df_dict,
        "dims_for_embedding_dict": dims_for_embedding_dict,
    }
