# Development Guide

This document is for contributors and maintainers of UMAP-DEA. It focuses on local setup, project structure, development workflow, and repository-specific behavior.

For end-user instructions, see [README.md](README.md).

## Development goals

The repository currently serves two related purposes:
- reproducible simulation runs from command-line scripts,
- exploration and validation through tests and notebooks.

When making changes, keep both in mind:
- command-line workflows should remain simple,
- tests should reflect the implemented behavior,
- notebooks may lag behind the main code and should not be treated as the source of truth.

## Environment setup

### Python version

The repository includes [.python-version](.python-version) and currently targets Python `3.12.0` for local development.

The package metadata in [pyproject.toml](pyproject.toml) declares `>=3.9`, but if you want to match the maintained local workflow exactly, use Python `3.12.0`.

### Recommended setup: `pyenv` + `venv`

If you use `pyenv`, the repository includes [scripts/setup_dev.sh](scripts/setup_dev.sh), which:
- checks for `pyenv`,
- installs the configured Python version if needed,
- creates a standard `venv` directory named `.venv/` using the pyenv-managed Python.

Typical flow:

```bash
bash scripts/setup_dev.sh
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional notebook dependencies:

```bash
pip install -e ".[dev,jupyter]"
```

### Alternative setup: standard `venv`

If you do not use `pyenv`:

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -e ".[dev]"
```

## Dependency sources

There are two dependency definitions in this repository:

- [pyproject.toml](pyproject.toml): canonical package metadata and optional extras
- [requirements.txt](requirements.txt): simpler dependency list used in some environments

For development work, prefer [pyproject.toml](pyproject.toml).

Available optional dependency groups:
- `dev`: pytest, coverage, formatting, linting, mypy
- `jupyter`: notebook tooling

Examples:

```bash
pip install -e .
pip install -e ".[dev]"
pip install -e ".[dev,jupyter]"
```

## Running the project

### Main simulation entry point

Use [scripts/run_sim.py](scripts/run_sim.py) to execute a simulation batch from a JSON configuration:

```bash
python scripts/run_sim.py --config config.json
```

### Grid search

Use [scripts/run_grid_search.py](scripts/run_grid_search.py) to iterate over parameter combinations and launch repeated simulation runs. The script accepts any config key in the grid — not just `N` and `n`.

```bash
# Default grid over N and n
python scripts/run_grid_search.py

# Custom grid over N and n (overrides defaults)
python scripts/run_grid_search.py --param-grid '{"N": [50, 100], "n": [50, 100]}'

# Grid over N and rts (varying returns to scale)
python scripts/run_grid_search.py --param-grid '{"N": [50, 100, 200], "rts": ["crs", "vrs"]}'

# Grid over N, n, and rts (3-way grid, all combinations)
python scripts/run_grid_search.py --param-grid '{"N": [20, 50], "n": [20, 50], "rts": ["crs", "vrs"]}'

# Verbose mode to see per-simulation logs
python scripts/run_grid_search.py --param-grid '{"N": [50, 100], "rts": ["crs", "vrs"]}' --verbose
```

After installation, the grid-search runner is also available as the console script `umap-dea-run`:

```bash
umap-dea-run --param-grid '{"N": [50, 100], "n": [50, 100]}'
```

Results are placed under `results_grid_search/` by default. Each simulation uses the base config from `config.json`, with only the grid parameters overridden.

### Configuration model

The main configuration dataclass lives in [umap_dea/config.py](umap_dea/config.py).

Current fields include:
- `N`, `M`, `n`
- `alpha_1`, `gamma`, `sigma_u`
- `rts`, `orientation`
- `nr_simulations`, `seed`, `pca`
- `umap_n_neighbors`, `umap_min_dist`, `umap_metric`

Validation currently checks and resolves:
- `alpha_1` is a `float` or the string `"1/N"`
- `rts` is `crs` or `vrs`
- `orientation` is `input` or `output`

## Project architecture

### [umap_dea/dgp.py](umap_dea/dgp.py)

Implements the data-generating process.

Key responsibilities:
- generate normalized production coefficients,
- generate efficient outputs,
- generate inputs,
- inject inefficiency into observed outputs.

### [umap_dea/dim_red.py](umap_dea/dim_red.py)

Handles dimensionality reduction.

Important behavior:
- supports both UMAP and PCA,
- creates multiple embeddings for several target dimensions,
- always adds an `original` representation alongside reduced embeddings,
- shifts embeddings to non-negative values when needed.

The inclusion of `original` matters downstream because evaluation and summary outputs include it as a baseline.

### [umap_dea/dea.py](umap_dea/dea.py)

Wraps `dealib.dea.dea()` and computes efficiency scores for each embedding.

Important behavior:
- accepts `crs` and `vrs`,
- accepts `input` and `output` orientation,
- returns the library's score convention directly.

Do not assume input- and output-oriented scores share the same scale semantics.

### [umap_dea/eval.py](umap_dea/eval.py)

Builds the evaluation dataframe and computes:
- MAE
- Spearman correlation
- Pearson correlation
- Kendall's tau
- number of non-NaN observations

Current behavior to be aware of:
- the evaluation dataframe is built from `dims_for_embedding_dict`,
- rows may therefore include the `original` baseline,
- `dim_reduction_level` is the naming column used in exports and summaries.

### [scripts/run_sim.py](scripts/run_sim.py)

Coordinates the full workflow:
- data generation,
- dimensionality reduction,
- DEA,
- evaluation,
- CSV export,
- multiprocessing across simulation iterations.

The script also creates a `results/` directory on demand.

### [scripts/organize_results.py](scripts/organize_results.py)

Organizes flat CSV output files into a structured directory hierarchy. Run it after a batch of simulations to group results by gamma value and reduction method:

```bash
python scripts/organize_results.py
```

The script:
1. Recursively discovers all CSV files under `results/`,
2. Groups files by UUID and deduplicates identical copies by keeping the file with the most recent modification time,
3. Reads each `params_dict.csv` to extract `pca`, `umap_n_neighbors`, and `gamma`,
4. Moves files into the target structure:

```text
results/
├── gamma_0p5/
│   ├── pca_dea/              # pca=True, gamma=0.5
│   └── umap_dea/
│       ├── k_05/             # pca=False, gamma=0.5, umap_n_neighbors=5
│       ├── k_15/             # pca=False, gamma=0.5, umap_n_neighbors=15
│       └── ...
└── gamma_1/
    ├── pca_dea/              # pca=True, gamma=1.0
    └── umap_dea/
        ├── k_05/             # pca=False, gamma=1.0, umap_n_neighbors=5
        ├── k_15/             # pca=False, gamma=1.0, umap_n_neighbors=15
        └── ...
```

Gamma values are formatted as directory names (`0.5` → `0p5`, `1.0` → `1`). The `umap_n_neighbors` value is zero-padded to two digits.

### [scripts/sort_results.py](scripts/sort_results.py)

A simpler, non-recursive alternative to `organize_results.py` that sorts CSV files flat within `results/` into `results/pca_dea/` and `results/umap_dea/k_XX/` (without the gamma hierarchy):

```bash
python scripts/sort_results.py
```

### [scripts/latex/generate_all.sh](scripts/latex/generate_all.sh)

Orchestrator that runs the full LaTeX-generation pipeline:

```bash
bash scripts/latex/generate_all.sh
```

Optional flags:
- `--clean` — removes the `tex/` directory first
- `--skip-tables` — only generates appendix tables

The generated `.tex` files are written to `tex/` and can be included directly in a paper.

### [scripts/latex/generate_tables.py](scripts/latex/generate_tables.py)

Generates main result tables (primary paper tables) from organized results under `results/`.

### [scripts/latex/generate_appendix.py](scripts/latex/generate_appendix.py)

Generates appendix subsection `.tex` files from organized results.

### [scripts/latex/compare_vs_pca.py](scripts/latex/compare_vs_pca.py)

Generates head-to-head comparison tables of UMAP vs PCA.

### [scripts/latex/compare_vs_conventional.py](scripts/latex/compare_vs_conventional.py)

Generates comparison tables of dimension-reduced DEA vs conventional (full-dimension) DEA.

### [scripts/latex/_utils.py](scripts/latex/_utils.py)

Shared utilities used by the LaTeX generation scripts, including helper functions for loading organized results and formatting LaTeX output.

## Tests

Run tests with:

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=umap_dea --cov-report=html --cov-report=term
```

Pytest configuration lives in `[tool.pytest.ini_options]` within [pyproject.toml](pyproject.toml). The previously separate `pytest.ini` file has been removed in favor of consolidating all tool configuration in `pyproject.toml`.

### Testing philosophy

Prefer basic, behavior-focused tests.

Good tests in this repository usually:
- verify shapes, keys, and allowed ranges,
- check simple invariants,
- avoid over-specifying implementation details,
- follow the actual semantics of third-party libraries rather than forcing alternate conventions.

Be especially careful with DEA orientation semantics and with the `original` embedding baseline included by [umap_dea/dim_red.py](umap_dea/dim_red.py).

## Formatting and linting

Common commands:

```bash
black umap_dea tests
isort umap_dea tests
flake8 umap_dea tests
mypy umap_dea
```

If you use the [Makefile](Makefile), available targets include:
- `make install`
- `make install-dev`
- `make install-jupyter`
- `make test`
- `make test-cov`
- `make lint`
- `make format`
- `make mypy`

Note: some `Makefile` targets use Unix shell commands and may need adjustment on Windows.

## Working with notebooks

The notebooks in [experiments/](experiments) are useful for exploration, but they may not always be synchronized with the latest implementation.

Before relying on notebook logic:
- verify the equivalent code in [umap_dea/](umap_dea),
- confirm assumptions against tests,
- prefer script-based workflows for reproducible runs.

## Common pitfalls

### 1. Script locations

All runnable scripts live under `scripts/`. Use `python scripts/run_sim.py` (not `python run_sim.py`) and `python scripts/run_grid_search.py` (or the `umap-dea-run` console script).

### 2. Assuming all evaluation rows are reduced embeddings

They are not. The `original` baseline is intentionally included in the embedding pipeline.

### 3. Assuming output-oriented DEA scores must match input-oriented scaling

They do not necessarily. The wrapper currently preserves `dealib` semantics.

### 4. Treating notebooks as authoritative

The main implementation lives in [umap_dea/](umap_dea) and the runner scripts in [scripts/](scripts).

## Suggested workflow for contributors

1. Create or activate the environment.
2. Install `.[dev]` or `.[dev,jupyter]`.
3. Make focused changes.
4. Run tests.
5. Run formatting and lint checks if relevant.
6. Update documentation when behavior or workflow changes.

## When updating documentation

Use this rule of thumb:
- [README.md](README.md): explain what the project is, how to install it, how to run it, and what outputs to expect.
- [DEVELOPMENT.md](DEVELOPMENT.md): explain how to work on the codebase, validate changes, and avoid repository-specific pitfalls.
