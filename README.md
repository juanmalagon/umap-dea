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
├── run_sim.py               Main simulation entry point
├── run_grid_search.py       Parameter-grid runner
├── umap_dea/
│   ├── config.py            Simulation configuration dataclass
│   ├── dgp.py               Data-generating process
│   ├── dim_red.py           UMAP/PCA dimensionality reduction helpers
│   ├── dea.py               DEA wrapper around dealib
│   └── eval.py              Evaluation metrics and result dataframe creation
├── tests/                   Unit tests
├── experiments/             Notebooks and experimental scripts
├── requirements.txt         Simple dependency list
├── pyproject.toml           Package metadata and optional dependency groups
├── pytest.ini               Active pytest configuration
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
python run_sim.py --config config.json
```

If `--config` is omitted, the script defaults to `config.json`.

### 3. Review outputs

The simulation creates a `results/` directory if it does not already exist and writes files such as:

- `params_dict_<UUID>.csv`: parameters used for the run
- `evaluation_df_<UUID>.csv`: evaluation results for each simulation iteration
- `summary_df_<UUID>.csv`: aggregated summary statistics
- `errors_list_<UUID>.csv`: iterations that failed

## Grid search

To run a simple grid search over selected parameters:

```bash
python run_grid_search.py
```

By default, the script varies `N` and `n` over a predefined grid and launches `run_sim.py` for each configuration.

## Interpreting results

The exported evaluation files include metrics such as:
- `mae`
- `spearmanr`
- `pearsonr`
- `kendalltau`
- `nr_non_nan`

The `evaluation_df` includes entries for reduced embeddings and for the `original` input space, which is used as a baseline.

For DEA orientation:
- input-oriented scores are expected to lie in the usual efficiency range,
- output-oriented scores come directly from `dealib` and follow that library's output-oriented convention.

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

Note: this repository uses [pytest.ini](pytest.ini) as the active pytest configuration file.

## Notebooks and experiments

The [experiments/](experiments) folder contains notebooks and exploratory scripts used during development and analysis. These are useful for inspection and experimentation, but the main reproducible workflow is driven by [run_sim.py](run_sim.py) and [run_grid_search.py](run_grid_search.py).

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