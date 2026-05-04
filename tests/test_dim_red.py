import pytest
import numpy as np
from umap_dea import dim_red


class TestGetDimsForEmbedding:
    """Test get_dims_for_embedding function."""

    def test_dims_calculation(self):
        """Test that embedding dimensions are correctly calculated."""
        x = np.random.rand(100, 20)
        dims_dict = dim_red.get_dims_for_embedding(x)

        assert dims_dict["half"] == 10  # 20 / 2
        assert dims_dict["sqrt"] == int(np.sqrt(20))
        assert dims_dict["log"] == int(np.log(20))
        assert dims_dict["ten_percent"] == 2  # 20 * 0.1

    def test_dims_adjustment_for_small_n_samples(self):
        """Test that dimensions are adjusted when n_components >= n_samples."""
        x = np.random.rand(10, 20)  # Small number of samples
        dims_dict = dim_red.get_dims_for_embedding(x)

        # All dims should be adjusted to at most n_samples - 2
        for dim in dims_dict.values():
            assert dim < x.shape[0]

    def test_dims_dict_keys(self):
        """Test that all expected keys are present."""
        x = np.random.rand(50, 15)
        dims_dict = dim_red.get_dims_for_embedding(x)

        expected_keys = {"half", "sqrt", "log", "ten_percent"}
        assert set(dims_dict.keys()) == expected_keys


class TestReduceDims:
    """Test reduce_dims function."""

    def test_reduce_dims_output_shape(self, sample_data):
        """Test that reduce_dims produces correct output shape."""
        x = sample_data["x"]
        n_components = 5

        result = dim_red.reduce_dims(x, n_components=n_components, seed=42)

        assert result.shape == (x.shape[0], n_components)

    def test_reduce_dims_non_negative(self, sample_data):
        """Test that output values are non-negative."""
        x = sample_data["x"]
        result = dim_red.reduce_dims(x, n_components=2, seed=42)

        assert np.all(result >= 0)

    def test_reduce_dims_with_different_metrics(self, sample_data):
        """Test reduce_dims with different metrics."""
        x = sample_data["x"]
        metrics = ["euclidean", "cosine"]

        for metric in metrics:
            result = dim_red.reduce_dims(
                x, n_components=2, metric=metric, seed=42
            )
            assert result.shape == (x.shape[0], 2)
            assert np.all(result >= 0)

    def test_reduce_dims_with_different_n_neighbors(self, sample_data):
        """Test reduce_dims with different n_neighbors values."""
        x = sample_data["x"]

        result1 = dim_red.reduce_dims(x, n_components=2, n_neighbors=10, seed=42)
        result2 = dim_red.reduce_dims(x, n_components=2, n_neighbors=20, seed=42)

        assert result1.shape == result2.shape
        # Results should differ but both be valid
        assert not np.allclose(result1, result2, rtol=1e-5)

    def test_reduce_dims_reproducibility(self, sample_data):
        """Test that reduce_dims is reproducible with same seed."""
        x = sample_data["x"]

        result1 = dim_red.reduce_dims(x, n_components=2, seed=42)
        result2 = dim_red.reduce_dims(x, n_components=2, seed=42)

        np.testing.assert_array_almost_equal(result1, result2)


class TestReduceDimensionsWithPCA:
    """Test reduce_dimensions_with_pca function."""

    def test_pca_output_shape(self, sample_data):
        """Test that PCA produces correct output shape."""
        x = sample_data["x"]
        d = 3

        result = dim_red.reduce_dimensions_with_pca(x, d=d, random_state=42)

        assert result.shape == (x.shape[0], d)

    def test_pca_non_negative(self, sample_data):
        """Test that output values are non-negative."""
        x = sample_data["x"]
        result = dim_red.reduce_dimensions_with_pca(x, d=3, random_state=42)

        assert np.all(result >= 0)

    def test_pca_reproducibility(self, sample_data):
        """Test that PCA is reproducible with same seed."""
        x = sample_data["x"]

        result1 = dim_red.reduce_dimensions_with_pca(x, d=3, random_state=42)
        result2 = dim_red.reduce_dimensions_with_pca(x, d=3, random_state=42)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_pca_invalid_d(self, sample_data):
        """Test that requesting more dimensions than available raises error."""
        x = sample_data["x"]

        with pytest.raises(ValueError, match="Cannot reduce to"):
            dim_red.reduce_dimensions_with_pca(x, d=x.shape[1] + 1)


class TestCreateEmbeddings:
    """Test create_embeddings function."""

    def test_create_embeddings_umap(self, sample_data):
        """Test creating embeddings with UMAP."""
        x = sample_data["x"]
        embeddings = dim_red.create_embeddings(x, seed=42, pca=False)

        assert "embeddings_df_dict" in embeddings
        assert "dims_for_embedding_dict" in embeddings
        assert "original" in embeddings["embeddings_df_dict"]

    def test_create_embeddings_pca(self, sample_data):
        """Test creating embeddings with PCA."""
        x = sample_data["x"]
        embeddings = dim_red.create_embeddings(x, seed=42, pca=True)

        assert "embeddings_df_dict" in embeddings
        assert "dims_for_embedding_dict" in embeddings

    def test_create_embeddings_with_umap_params(self, sample_data):
        """Test creating embeddings with custom UMAP parameters."""
        x = sample_data["x"]
        embeddings = dim_red.create_embeddings(
            x,
            seed=42,
            pca=False,
            umap_n_neighbors=10,
            umap_min_dist=0.05,
            umap_metric="cosine",
        )

        assert "embeddings_df_dict" in embeddings
        assert "dims_for_embedding_dict" in embeddings

    def test_create_embeddings_all_dimensions_valid(self, sample_data):
        """Test that all created embeddings have valid shapes."""
        x = sample_data["x"]
        embeddings = dim_red.create_embeddings(x, seed=42, pca=False)

        embeddings_dict = embeddings["embeddings_df_dict"]
        dims_dict = embeddings["dims_for_embedding_dict"]

        for key, embedding in embeddings_dict.items():
            if key != "original":
                assert embedding.shape[0] == x.shape[0]
                assert embedding.shape[1] == dims_dict[key]
                assert np.all(embedding >= 0)
