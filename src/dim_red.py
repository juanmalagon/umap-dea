import numpy as np
import umap
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def get_dims_for_embedding(x: np.ndarray) -> dict:
    """
    Returns a dictionary with the number of dimensions to use for each
    embedding.
    The numbers of dimensions are calculated based on the dimensions of the
    input array: half, square root, logarithm and 10%.
    """
    dims_for_embedding_dict = {
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
    """
    Reduces the number of dimensions of the input array
    using UMAP.
    """
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
    )
    u = fit.fit_transform(x)
    # Linear translation to make all values non-negative
    if (u < 0).any():  # type: ignore
        u = u - u.min()  # type: ignore
    print(f"Shape of the embedding: {u.shape}")  # type: ignore
    return u  # type: ignore


def reduce_dimensions_with_pca(X: np.ndarray, d: int, random_state: int = None, verbose: bool = False) -> np.ndarray:
    """
    Reduce the dimensionality of input data using PCA with standardization.

    Parameters:
    -----------
    X : np.ndarray
        Input data array of shape (n, N) where n is number of samples and N is original number of features
    d : int
        Number of dimensions to reduce to (must be <= N)
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, optional
        Whether to print information about explained variance

    Returns:
    --------
    np.ndarray
        Reduced data array of shape (n, d)
    """
    if d > X.shape[1]:
        raise ValueError(f"Cannot reduce to {d} dimensions when input has only {X.shape[1]} features")

    # Step 1: Standardize the data (z-score normalization)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Step 2: Apply PCA
    pca = PCA(n_components=d, random_state=random_state)
    X_reduced = pca.fit_transform(X_std)

    # Linear translation to make all values non-negative
    if (X_reduced < 0).any():  # type: ignore
        X_reduced = X_reduced - X_reduced.min()  # type: ignore

    if verbose:
        # Print standardization info
        print(f"Standardization complete (mean={scaler.mean_.round(2)}, std={scaler.scale_.round(2)})")

        # Print explained variance information
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance ratio by each component: {explained_variance.round(4)}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")

    # Print shape information
    print(f"Shape transformed from {X.shape} to {X_reduced.shape}")

    return X_reduced


def create_embeddings(x: np.ndarray, seed: int = 42, pca=False) -> dict:
    """
    Creates embeddings with different dimensions.
    """
    print(f"Original shape: {x.shape}")
    dims_for_embedding_dict = get_dims_for_embedding(x)
    embeddings_df_dict = {}
    # Creating embeddings with different dimensions according to the dictionary dims_for_embedding_dict
    if pca:
        print("Using PCA for dimensionality reduction")
        for k, v in dims_for_embedding_dict.items():
            print(f"Creating embedding with {v} dimensions ({k})")
            embeddings_df_dict[k] = reduce_dimensions_with_pca(x, d=v, random_state=seed, verbose=False)
    else:
        print("Using UMAP for dimensionality reduction")
        for k, v in dims_for_embedding_dict.items():
            print(f"Creating embedding with {v} dimensions ({k})")
            embeddings_df_dict[k] = reduce_dims(x, n_components=v, seed=seed)
    # Adding the original array to the dictionary
    dims_for_embedding_dict["original"] = x.shape[1]
    embeddings_df_dict["original"] = x
    return {
        "embeddings_df_dict": embeddings_df_dict,
        "dims_for_embedding_dict": dims_for_embedding_dict,
    }
