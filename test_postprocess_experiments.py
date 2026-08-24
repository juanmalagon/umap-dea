from scripts.postprocess_experiments import comparison_runs_for_case, generate_all_plots


def _summary_rows_for_run(run):
    return [
        {
            "algorithm": run["algo"],
            "N": run["N"],
            "n": run["n"],
            "rts": run["rts"],
            "gamma": run["gamma"],
            "umap_n_neighbors": run["umap_n_neighbors"],
            "nr_simulations": run["nr_simulations"],
            "dim_reduction_level": "original",
            "dims": run["N"],
            "kendalltau_mean": 0.2,
            "kendalltau_std": 0.01,
            "seed": 42,
        },
        {
            "algorithm": run["algo"],
            "N": run["N"],
            "n": run["n"],
            "rts": run["rts"],
            "gamma": run["gamma"],
            "umap_n_neighbors": run["umap_n_neighbors"],
            "nr_simulations": run["nr_simulations"],
            "dim_reduction_level": "sqrt",
            "dims": 3,
            "kendalltau_mean": 0.1,
            "kendalltau_std": 0.02,
            "seed": 42,
        },
    ]


def test_comparison_runs_for_case_labels_collapsed_duplicate_k_rules():
    base_config = {"rts": "vrs", "gamma": 0.5, "nr_simulations": 1000}

    runs = comparison_runs_for_case(base_config, n_inputs=10, n_dmus=10)

    assert runs == [
        {
            "N": 10,
            "n": 10,
            "rts": "vrs",
            "gamma": 0.5,
            "nr_simulations": 1000,
            "algo": "PCA-DEA",
            "umap_n_neighbors": 3,
        },
        {
            "N": 10,
            "n": 10,
            "rts": "vrs",
            "gamma": 0.5,
            "nr_simulations": 1000,
            "algo": "UMAP-DEA",
            "umap_n_neighbors": 3,
            "k_label": "log2(n) = sqrt(n)",
        },
        {
            "N": 10,
            "n": 10,
            "rts": "vrs",
            "gamma": 0.5,
            "nr_simulations": 1000,
            "algo": "UMAP-DEA",
            "umap_n_neighbors": 5,
            "k_label": "n/2",
        },
    ]


def test_comparison_runs_for_case_labels_three_distinct_k_rules():
    base_config = {"rts": "vrs", "gamma": 0.5, "nr_simulations": 1000}

    runs = comparison_runs_for_case(base_config, n_inputs=50, n_dmus=500)

    assert [run["umap_n_neighbors"] for run in runs] == [3, 8, 22, 250]
    assert [run.get("k_label") for run in runs] == [
        None,
        "log2(n)",
        "sqrt(n)",
        "n/2",
    ]


def test_three_run_collapsed_case_plot_saves_file(tmp_path):
    import pandas as pd

    from experiments.compare_four_runs_plot import plot_comparison

    base_config = {"rts": "vrs", "gamma": 0.5, "nr_simulations": 1000}
    runs = comparison_runs_for_case(base_config, n_inputs=10, n_dmus=10)
    df = pd.DataFrame(row for run in runs for row in _summary_rows_for_run(run))

    plot_comparison(
        df,
        runs,
        metric_names=["Kendall"],
        show_std=False,
        output_dir=tmp_path,
        show=False,
    )

    assert len(list(tmp_path.glob("Kendall__bar__*.png"))) == 1


def test_algorithm_column_handles_string_and_boolean_pca_values():
    import pandas as pd

    from experiments.compare_four_runs_data import add_algorithm_column

    df = pd.DataFrame({"pca": [True, False, "True", "False", " true ", " false "]})

    result = add_algorithm_column(df)

    assert result["algorithm"].tolist() == [
        "PCA-DEA",
        "UMAP-DEA",
        "PCA-DEA",
        "UMAP-DEA",
        "PCA-DEA",
        "UMAP-DEA",
    ]


def test_generate_all_plots_skips_completely_missing_cases(tmp_path):
    import pandas as pd

    base_config = {"rts": "vrs", "gamma": 0.5, "nr_simulations": 1000}
    runs = comparison_runs_for_case(base_config, n_inputs=10, n_dmus=10)
    df = pd.DataFrame(row for run in runs for row in _summary_rows_for_run(run))

    generate_all_plots(
        df=df,
        base_config=base_config,
        plots_root=tmp_path,
        metric_names=["Kendall"],
        show_std=False,
        allow_missing=False,
    )

    assert len(list(tmp_path.rglob("Kendall__bar__*.png"))) == 1