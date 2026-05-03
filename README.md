# Dimensionality Reduction in DEA using UMAP

This repository contains the code and resources for the paper **"Dimensionality Reduction in Data Envelopment Analysis using Uniform Manifold Approximation and Projection"**. The study investigates the integration of UMAP (Uniform Manifold Approximation and Projection) with Data Envelopment Analysis (DEA) to mitigate the curse of dimensionality in efficiency estimation. The framework is validated through extensive Monte Carlo simulations.

---

## 📁 Project Structure

```
.
├── config.json            # JSON configuration file for simulations
├── src/
│   ├── config.py          # Simulation configuration dataclass
│   ├── dgp.py             # Data Generating Process (DGP) for production data
│   ├── dim_red.py         # Dimensionality reduction (UMAP/PCA) utilities
│   ├── dea.py             # DEA efficiency computation wrapper
│   ├── eval.py            # Evaluation metrics (MAE, correlations)
│   └── run_sim.py         # Main simulation runner
├── results/               # Output directory for simulation results
├── experiments/           # Experimental notebooks and legacy scripts
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a Simulation

Configure your simulation parameters in `config.json` (an example is provided). The file should contain a JSON object with keys matching the `SimulationConfig` dataclass fields.

Example `config.json`:
```json
{
    "N": 100,
    "M": 1,
    "n": 200,
    "alpha_1": 0.25,
    "gamma": 1.0,
    "sigma_u": 0.1,
    "rts": "crs",
    "orientation": "input",
    "nr_simulations": 1000,
    "seed": 42,
    "pca": false
}
```

Run the simulation:
```bash
python run_sim.py --config config.json
```

You can specify a different config file path with `--config <path>`. If omitted, it defaults to `config.json`.

---

## 📊 Outputs

The simulation generates the following files in `results/`:

- `params_dict_<UUID>.csv` — Simulation parameters
- `evaluation_df_<UUID>.csv` — Raw evaluation metrics per replication
- `summary_df_<UUID>.csv` — Aggregated results (mean ± std)
- `errors_list_<UUID>.csv` — List of failed iterations (if any)

Metrics include:
- Mean Absolute Error (MAE)
- Spearman’s rank correlation
- Pearson correlation
- Kendall’s τ

---

## 🔧 Key Functions

### Data Generation (`dgp.py`)
- `generate_data_dict()`: Generates input-output data using Cobb-Douglas production technology with inefficiency noise.

### Dimensionality Reduction (`dim_red.py`)
- `reduce_dims()`: Applies UMAP to reduce input dimensions.
- `reduce_dimensions_with_pca()`: Alternative PCA-based reduction.
- `create_embeddings()`: Generates embeddings for multiple dimension levels.

### DEA Computation (`dea.py`)
- `calculate_dea_for_embeddings()`: Computes DEA efficiency scores for all embeddings.

### Evaluation (`eval.py`)
- `create_evaluation_df()`: Computes metrics comparing estimated vs. true efficiencies.

---

## 🧪 Example Use Cases

1. **Compare UMAP vs. PCA**:
   ```python
   config = SimulationConfig(..., pca=True)
   ```

2. **Test different returns-to-scale assumptions**:
   ```python
   config = SimulationConfig(..., rts='vrs')
   ```

3. **Validate under extreme dimensionality**:
   ```python
   config = SimulationConfig(N=200, n=20, ...)
   ```

---

## � Citation

If you use this code or findings in your work, please cite:

```bibtex
@article{malagon2025dimensionality,
  title={Dimensionality Reduction in Data Envelopment Analysis using Uniform Manifold Approximation and Projection},
  author={Malagon J, Grigoriev A, Haelermans C},
  year={2025}
}
```

---

## 📬 Contact

For questions or collaborations, please contact malagon@alumni.harvard.edu or open an issue in this repository.

---

## 📄 License

This project is licensed under the Apache License, Version 2.0. See `LICENSE` for details.

---

**Note**: This code is provided as supplementary material for the associated paper. Results may vary based on hardware and software configurations.