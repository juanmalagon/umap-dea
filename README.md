# UMAP-DEA

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Package](https://img.shields.io/badge/package-umap__dea-informational)](pyproject.toml)

UMAP-DEA is a simulation framework for studying dimensionality reduction in Data Envelopment Analysis (DEA), with a focus on UMAP and PCA as preprocessing steps before efficiency estimation.

This repository contains:
- simulation code for generating synthetic production data,
- dimensionality-reduction pipelines,
- DEA wrappers based on `dealib`,
- evaluation utilities for comparing estimated and theoretical efficiencies,
- tests and exploratory notebooks.

## What this project does

At a high level, a simulation run follows this workflow:

1. Generate synthetic inputs and outputs from a data-generating process.
2. Create lower-dimensional representations of the input space.
3. Run DEA on each embedding, plus the original input space.
4. Compare estimated efficiency scores against the theoretical design efficiency.
5. Export raw and aggregated results as CSV files.

## Repository layout

```text
.
├── config.json              Default simulation configuration
├── pyproject.toml           Package metadata, dependencies, and tool config
├── requirements.txt         Simple dependency list (mirrors pyproject.toml)
├── Makefile                 Common development commands
├── umap_dea/
│   ├── config.py            Simulation configuration dataclass
│   ├── dgp.py               Data-generating process
│   ├── dim_red.py           UMAP/PCA dimensionality reduction helpers
│   ├── dea.py               DEA wrapper around dealib
│   └── eval.py              Evaluation metrics and result dataframe creation
├── scripts/
│   ├── run_sim.py           Main simulation entry point
│   ├── run_grid_search.py   Parameter-grid runner (also exposed as `umap-dea-run`)
│   ├── find_best_dim_reduction.py    Best dimensionality reduction analysis
│   ├── organize_results.py  Organize results by gamma and reduction method
│   ├── sort_results.py      Sort results into pca_dea/ and umap_dea/ subdirectories
│   ├── setup_dev.sh         Developer environment setup helper
│   └── latex/
│       ├── generate_all.sh            Orchestrator for all LaTeX generation scripts
│       ├── generate_tables.py         Generate LaTeX tables from results
│       ├── generate_appendix.py       Generate appendix subsection .tex files
│       ├── compare_vs_pca.py          UMAP vs PCA head-to-head comparison tables
│       └── compare_vs_conventional.py   Dimension-reduced vs conventional DEA tables
├── tests/                   Unit tests
├── experiments/             Notebooks and exploratory scripts
├── results/                 Simulation output, organized by gamma and method
├── results_grid_search/     Grid search output (flat CSV directory)
├── tex/                     Generated LaTeX tables
└── DEVELOPMENT.md           Developer-oriented setup and workflow guide
```

## Installation

You can install the project in either of these ways.

### Option 1: install from `pyproject.toml`

This is the better option if you want an editable install and optional extras.

```bash
pip install -e .
```

For notebook support:

```bash
pip install -e ".[jupyter]"
```

### Option 2: install from `requirements.txt`

This is a simpler option if you only want the runtime and test dependencies listed there.

```bash
pip install -r requirements.txt
```

## Quick start

### 1. Review the configuration

The default configuration lives in `config.json`.

Example:

```json
{
   "N": 200,
   "M": 1,
   "n": 200,
   "alpha_1": 0.25,
   "gamma": 1.0,
   "sigma_u": 0.1,
   "rts": "crs",
   "orientation": "input",
   "nr_simulations": 1000,
   "seed": 42,
   "pca": false,
   "umap_n_neighbors": 15,
   "umap_min_dist": 0.1,
   "umap_metric": "euclidean"
}
```

Important parameters:
- `N`: number of input variables
- `M`: number of output variables
- `n`: number of decision-making units
- `alpha_1`: first input elasticity, either a float or the string `"1/N"`
- `rts`: returns to scale, `crs` or `vrs`
- `orientation`: `input` or `output`
- `pca`: if `true`, use PCA; otherwise use UMAP
- `umap_n_neighbors`, `umap_min_dist`, `umap_metric`: UMAP hyperparameters

### 2. Run a simulation study

```bash
python scripts/run_sim.py --config config.json
```

If `--config` is omitted, the script defaults to `config.json` at the project root.

The grid-search runner is also available as a console script after installation:

```bash
umap-dea-run --param-grid '{"N": [20, 50], "n": [20, 50]}'
```

### 3. Review outputs

The simulation creates a `results/` directory if it does not already exist and writes files such as:

- `params_dict_<UUID>.csv`: parameters used for the run
- `evaluation_df_<UUID>.csv`: evaluation results for each simulation iteration
- `summary_df_<UUID>.csv`: aggregated summary statistics
- `errors_list_<UUID>.csv`: iterations that failed

## Grid search

Run a parameter grid search with `run_grid_search.py`. The script takes the base configuration from `config.json` and overrides the grid parameters for each combination.

```bash
# Default grid over N and n
python scripts/run_grid_search.py

# Custom grid over any config parameters (overrides defaults)
python scripts/run_grid_search.py --param-grid '{"N": [50, 100], "n": [50, 100]}'

# Grid over N and returns-to-scale (rts)
python scripts/run_grid_search.py --param-grid '{"N": [50, 100, 200], "rts": ["crs", "vrs"]}'

# 3-way grid over N, n, and orientation
python scripts/run_grid_search.py --param-grid '{"N": [20, 50], "n": [20, 50], "orientation": ["input", "output"]}'
```

Or use the installed console script:

```bash
umap-dea-run --param-grid '{"N": [50, 100], "n": [50, 100]}'
```

You can grid search over **any** config key — `N`, `n`, `rts`, `orientation`, `alpha_1`, `gamma`, `sigma_u`, `umap_n_neighbors`, `umap_min_dist`, `umap_metric`, etc. All combinations of the provided values are run.

Without `--param-grid`, the default grid is `N ∈ [20, 50, 100, 200]` and `n ∈ [20, 50, 100, 200]`.

Results are placed in `results_grid_search/`.

## Results organization

After running simulations, the output files are written to `results/`. When you have simulations spanning multiple gamma values and reduction methods, you can organize them with `scripts/organize_results.py`:

```bash
python scripts/organize_results.py
```

This script:
1. Recursively discovers all CSV files under `results/`,
2. Groups files by UUID and deduplicates identical copies,
3. Reads each `params_dict.csv` to extract `pca`, `umap_n_neighbors`, and `gamma`,
4. Moves files into the target structure:

```text
results/
├── gamma_0p5/
│   ├── pca_dea/              # PCA-based runs with gamma = 0.5
│   └── umap_dea/
│       ├── k_05/             # UMAP runs, umap_n_neighbors = 5
│       ├── k_15/             # UMAP runs, umap_n_neighbors = 15
│       └── ...
└── gamma_1/
    ├── pca_dea/              # PCA-based runs with gamma = 1.0
    └── umap_dea/
        ├── k_05/
        ├── k_15/
        └── ...
```

Gamma values are formatted as directory names (e.g., `0.5` → `0p5`, `1.0` → `1`).

A simpler (non-recursive, flat) alternative is `scripts/sort_results.py`, which sorts files into `results/pca_dea/` and `results/umap_dea/k_XX/` without the gamma hierarchy.

## Interpreting results

Each simulation run writes four CSV files to `results/`:

| File | Content |
|---|---|
| `params_dict_<UUID>.csv` | Parameters used for this run |
| `evaluation_df_<UUID>.csv` | Per-iteration, per-embedding metrics |
| `summary_df_<UUID>.csv` | Mean, standard deviation, and warning counts aggregated by embedding and dimensionality |
| `errors_list_<UUID>.csv` | Iteration indices that failed with an exception |

### `evaluation_df_<UUID>.csv`

One row per embedding (`original` baseline + reduced embeddings) per simulation iteration.

| Column | Type | Description |
|---|---|---|
| `dim_reduction_level` | `str` | Embedding label (e.g. `original`, `embedding`) |
| `dims` | `int` | Number of dimensions in the embedding |
| `iteration` | `int` | Simulation iteration (0-indexed) |
| `mae` | `float` | Mean absolute error between estimated and design efficiency scores |
| `spearmanr` | `float` | Spearman rank correlation; `NaN` when undefined (see below) |
| `spearmanr_warning` | `bool` | `True` if Spearman correlation was undefined for this embedding |
| `pearsonr` | `float` | Pearson correlation |
| `kendalltau` | `float` | Kendall's τ; `NaN` when undefined |
| `kendalltau_warning` | `bool` | `True` if Kendall's τ was undefined for this embedding |
| `nr_non_nan` | `int` | Number of DMUs with valid (non-NaN) DEA scores |
| `nr_efficient` | `int` | Number of DMUs identified as efficient (score within 1e-6 of 1.0) |
| `prop_efficient` | `float` | Proportion of efficient DMUs among those with valid scores; `NaN` if no valid scores exist |

### `summary_df_<UUID>.csv`

Aggregated over all simulation iterations. Each row is one `(dim_reduction_level, dims)` group.

| Column | Type | Description |
|---|---|---|
| `dim_reduction_level` | `str` | Embedding label |
| `dims` | `int` | Number of dimensions in the embedding |
| `mae_mean` / `mae_std` | `float` | Mean and standard deviation of MAE |
| `spearmanr_mean` / `spearmanr_std` | `float` | Mean and standard deviation of Spearman's r (NaNs excluded) |
| `pearsonr_mean` / `pearsonr_std` | `float` | Mean and standard deviation of Pearson's r |
| `kendalltau_mean` / `kendalltau_std` | `float` | Mean and standard deviation of Kendall's τ (NaNs excluded) |
| `nr_efficient_mean` / `nr_efficient_std` | `float` | Mean and standard deviation of the efficient DMU count |
| `prop_efficient_mean` / `prop_efficient_std` | `float` | Mean and standard deviation of the efficient DMU proportion |
| `nr_non_nan_mean` / `nr_non_nan_std` | `float` | Mean and standard deviation of the valid DMU count |
| `spearmanr_warning_count` | `int` | Number of iterations with an undefined Spearman correlation |
| `kendalltau_warning_count` | `int` | Number of iterations with an undefined Kendall's τ |

### When are correlations undefined?

Spearman and Kendall correlations are set to `NaN` (and their corresponding warning flags to `True`) when:

- an efficiency-score array is constant (all DMUs are deemed equally efficient),
- fewer than 3 valid observations remain after dropping NaNs,
- a pair of arrays has fewer than 2 unique values in either array.

This happens, for example, when a large fraction of DMUs lands on the efficiency frontier, producing scores that are all very close or identical to 1.0. In those cases, rank correlation is not meaningful, and the summary statistics exclude those `NaN` entries from the computed mean and standard deviation.

### DEA orientation

- Input-oriented scores follow the usual DEA efficiency convention (≤ 1).
- Output-oriented scores follow `dealib`'s convention and use a different scale; do not compare them directly with input-oriented scores.

## Generating LaTeX tables

The `scripts/latex/` pipeline produces publication-ready LaTeX tables from organized results. After running simulations and organizing results with `scripts/organize_results.py`, run:

```bash
bash scripts/latex/generate_all.sh
```

This orchestrator script generates the full set of LaTeX output files with a single command. It accepts optional flags:

| Flag | Description |
|---|---|
| `--clean` | Remove the `tex/` directory before generating |
| `--skip-tables` | Skip main table generation (appendix only) |

The generated `.tex` files are written to the `tex/` directory and can be included directly in a paper or report.

The pipeline consists of several scripts:

- `generate_tables.py` — main result tables (primary paper tables)
- `generate_appendix.py` — appendix subsection tables
- `compare_vs_pca.py` — head-to-head comparison tables (UMAP vs PCA)
- `compare_vs_conventional.py` — dimension-reduced DEA vs conventional DEA

## Main modules

- `umap_dea/dgp.py`: synthetic data generation
- `umap_dea/dim_red.py`: UMAP and PCA embeddings
- `umap_dea/dea.py`: DEA score computation for all embeddings
- `umap_dea/eval.py`: comparison of estimated scores against design efficiency
- `umap_dea/config.py`: simulation configuration container and validation

## Testing

Run the test suite with:

```bash
pytest
```

Pytest configuration is defined in `[tool.pytest.ini_options]` within [pyproject.toml](pyproject.toml).

## Notebooks and experiments

The [experiments/](experiments) folder contains notebooks and exploratory scripts used during development and analysis. These are useful for inspection and experimentation, but the main reproducible workflow is driven by [scripts/run_sim.py](scripts/run_sim.py) and [scripts/run_grid_search.py](scripts/run_grid_search.py).

## Development

If you want to contribute or work on the codebase itself, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Citation

If you use this repository in academic work, please cite the associated paper or repository as appropriate.

```bibtex
@article{malagon2025dimensionality,
   title={Dimensionality Reduction in Data Envelopment Analysis using Uniform Manifold Approximation and Projection},
   author={Malagon J, Grigoriev A, Haelermans C},
   year={2025}
}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).