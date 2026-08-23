from scripts.run_all_experiments import (
    EXPERIMENT_CASES,
    build_experiment_configs,
    k_options_for_n,
    results_dir_for,
)


def test_build_experiment_configs_covers_each_pca_and_umap_case():
    base_config = {
        "N": 1,
        "n": 1,
        "orientation": "input",
        "pca": False,
        "umap_n_neighbors": 1,
    }

    configs = build_experiment_configs(base_config, "output")

    expected_count = len(EXPERIMENT_CASES) + sum(
        len(neighborhood_values) for _, _, neighborhood_values in EXPERIMENT_CASES
    )
    assert len(configs) == expected_count == 62
    assert {config["orientation"] for config in configs} == {"output"}
    assert sum(config["pca"] for config in configs) == len(EXPERIMENT_CASES)
    assert all(config["umap_n_neighbors"] < config["n"] for config in configs if not config["pca"])


def test_results_dir_for_matches_existing_result_layout(tmp_path):
    config = {
        "N": 10,
        "n": 25,
        "pca": False,
        "gamma": 0.5,
        "nr_simulations": 1000,
        "umap_n_neighbors": 12,
    }

    path = results_dir_for(config, tmp_path)

    assert (
        path == tmp_path / "nr_sim_1000" / "gamma_0p5" / "umap_dea" / "N_010" / "n_0025" / "k_012"
    )


def test_k_options_use_log2_sqrt_and_half_with_duplicates_collapsed():
    assert k_options_for_n(10) == [
        (3, "log2(n) = sqrt(n)"),
        (5, "n/2"),
    ]
    assert k_options_for_n(500) == [
        (8, "log2(n)"),
        (22, "sqrt(n)"),
        (250, "n/2"),
    ]


def test_k_options_match_historical_experiment_matrix():
    for _n_inputs, n_dmus, historical_k_values in EXPERIMENT_CASES:
        derived_k_values = tuple(k for k, _label in k_options_for_n(n_dmus))
        assert derived_k_values == historical_k_values
