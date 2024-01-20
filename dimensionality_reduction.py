import numpy as np
import umap
import warnings

warnings.filterwarnings('ignore')


def get_dims_for_embedding(x):
    dims_for_embedding_dict = {
        'half': int(x.shape[1]/2),
        'sqrt': int(np.sqrt(x.shape[1])),
        'log': int(np.log(x.shape[1])),
        'ten_percent': int(x.shape[1]*0.1),
    }
    # Correcting for spectral initialization in case
    # the number of dimensions is not less than x.shape[0]
    for k, v in dims_for_embedding_dict.items():
        if v >= x.shape[0]:
            dims_for_embedding_dict[k] = x.shape[0] - 2
    return dims_for_embedding_dict


def reduce_dims(
        x,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        seed=42):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed
    )
    u = fit.fit_transform(x)
    # Linear translation to make all values non-negative
    if (u < 0).any():  # type: ignore
        u = u - u.min()  # type: ignore
    print(f'Shape of the embedding: {u.shape}')  # type: ignore
    return u


def create_embeddings(x, seed):
    print(f'Original shape: {x.shape}')
    dims_for_embedding_dict = get_dims_for_embedding(x)
    embeddings_df_dict = {}
    for k, v in dims_for_embedding_dict.items():
        print(f'Creating embedding with {v} dimensions ({k})')
        embeddings_df_dict[k] = reduce_dims(x, n_components=v, seed=seed)
    embeddings_df_dict['original'] = x
    return {'embeddings_df_dict': embeddings_df_dict,
            'dims_for_embedding_dict': dims_for_embedding_dict}
