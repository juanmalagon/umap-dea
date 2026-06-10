import pytest
import numpy as np
from dealib.dea import RTS, Orientation
from umap_dea import dea


class TestCalculateDEAForEmbeddings:
    """Test calculate_dea_for_embeddings function."""

    def test_dea_output_structure(self, sample_data):
        """Test that DEA output has expected structure."""
        x = sample_data["x"]
        y = sample_data["y"]

        # Create a simple embeddings dictionary
        embeddings_df_dict = {
            "embedding1": x,
            "embedding2": x[:, : x.shape[1] // 2],
        }

        efficiency_scores = dea.calculate_dea_for_embeddings(
            embeddings_df_dict, y, rts="crs", orientation="input"
        )

        assert isinstance(efficiency_scores, dict)
        assert "embedding1" in efficiency_scores
        assert "embedding2" in efficiency_scores

    def test_dea_efficiency_scores_range(self, sample_data):
        """Test that DEA efficiency scores are in [0, 1] range."""
        x = sample_data["x"]
        y = sample_data["y"]

        embeddings_df_dict = {"embedding": x}

        efficiency_scores = dea.calculate_dea_for_embeddings(
            embeddings_df_dict, y, rts="crs", orientation="input"
        )

        scores = efficiency_scores["embedding"]
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_dea_with_vrs(self, sample_data):
        """Test DEA with variable returns to scale."""
        x = sample_data["x"]
        y = sample_data["y"]

        embeddings_df_dict = {"embedding": x}

        efficiency_scores = dea.calculate_dea_for_embeddings(
            embeddings_df_dict, y, rts="vrs", orientation="input"
        )

        scores = efficiency_scores["embedding"]
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_dea_with_output_orientation(self, sample_data):
        """Test DEA with output orientation."""
        x = sample_data["x"]
        y = sample_data["y"]

        embeddings_df_dict = {"embedding": x}

        efficiency_scores = dea.calculate_dea_for_embeddings(
            embeddings_df_dict, y, rts="crs", orientation="output"
        )

        scores = efficiency_scores["embedding"]
        # In output-oriented DEA, efficient DMUs score 1 and inefficient
        # DMUs score above 1 because the value is an output expansion factor.
        assert np.all(scores >= 1)

    def test_dea_invalid_rts(self, sample_data):
        """Test that invalid RTS raises error."""
        x = sample_data["x"]
        y = sample_data["y"]

        embeddings_df_dict = {"embedding": x}

        with pytest.raises(ValueError, match="rts must be either"):
            dea.calculate_dea_for_embeddings(
                embeddings_df_dict, y, rts="invalid", orientation="input"
            )

    def test_dea_invalid_orientation(self, sample_data):
        """Test that invalid orientation raises error."""
        x = sample_data["x"]
        y = sample_data["y"]

        embeddings_df_dict = {"embedding": x}

        with pytest.raises(ValueError, match="Orientation must be either"):
            dea.calculate_dea_for_embeddings(
                embeddings_df_dict, y, rts="crs", orientation="invalid"
            )

    def test_dea_multiple_embeddings_consistency(self, sample_data):
        """Test that multiple embeddings produce consistent structures."""
        x = sample_data["x"]
        y = sample_data["y"]

        # Create multiple embeddings of different sizes
        embeddings_df_dict = {
            "embedding1": x,
            "embedding2": x[:, : max(2, x.shape[1] // 2)],
            "embedding3": x[:, : max(1, x.shape[1] // 3)],
        }

        efficiency_scores = dea.calculate_dea_for_embeddings(
            embeddings_df_dict, y, rts="crs", orientation="input"
        )

        # All embeddings should have same number of DMUs
        n_dmus = efficiency_scores["embedding1"].shape[0]
        for scores in efficiency_scores.values():
            assert scores.shape[0] == n_dmus
