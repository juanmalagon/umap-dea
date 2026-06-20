"""
Aggregate all params_dict_*.csv and summary_df_*.csv pairs from the results/
folder into a single all_results.csv in the project root.

Each pair is matched by UUID and cross-joined (1 params row × N summary rows).
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_PATH = ROOT / "all_results.csv"

# Desired column order (config params first, then summary_df columns in original order)
COLUMN_ORDER = [
    # Config (params_dict)
    "seed",
    "alpha_1",
    "sigma_u",
    "gamma",
    "M",
    "rts",
    "orientation",
    "nr_simulations",
    "umap_min_dist",
    "umap_metric",
    "pca",
    "N",
    "n",
    "umap_n_neighbors",
    # Results (summary_df)
    "dim_reduction_level",
    "dims",
    "mae_mean",
    "mae_std",
    "spearmanr_mean",
    "spearmanr_std",
    "pearsonr_mean",
    "pearsonr_std",
    "kendalltau_mean",
    "kendalltau_std",
    "nr_efficient_mean",
    "nr_efficient_std",
    "prop_efficient_mean",
    "prop_efficient_std",
    "nr_non_nan_mean",
    "nr_non_nan_std",
    "spearmanr_warning_count",
    "kendalltau_warning_count",
]

UUID_RE = re.compile(r"^(?:params_dict|summary_df)_([a-f0-9-]+)\.csv$")


def extract_uuid(filename: str) -> str | None:
    m = UUID_RE.match(filename)
    return m.group(1) if m else None


def find_pairs(results_dir: Path) -> dict[str, dict[str, Path]]:
    """Return dict: uuid -> {'params': Path, 'summary': Path}."""
    pairs: dict[str, dict[str, Path]] = {}

    for fpath in sorted(results_dir.rglob("*.csv")):
        if not fpath.is_file():
            continue
        uuid = extract_uuid(fpath.name)
        if uuid is None:
            continue

        if uuid not in pairs:
            pairs[uuid] = {}

        if fpath.name.startswith("params_dict_"):
            pairs[uuid]["params"] = fpath
        elif fpath.name.startswith("summary_df_"):
            pairs[uuid]["summary"] = fpath

    # Keep only complete pairs
    return {u: d for u, d in pairs.items() if "params" in d and "summary" in d}


def read_params(path: Path) -> dict:
    """Read a params_dict CSV (header + 1 data row) and return a dict."""
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    # Ensure types match: boolean-like strings
    if "pca" in row:
        row["pca"] = str(row["pca"]).strip()
    return row


def main() -> None:
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    pairs = find_pairs(RESULTS_DIR)
    print(f"Found {len(pairs)} complete param-summary pairs.")

    if not pairs:
        print("No pairs found. Exiting.")
        return

    all_rows: list[dict] = []
    missing_cols: set[str] = set()
    skipped = 0

    for uuid, files in sorted(pairs.items()):
        try:
            params = read_params(files["params"])
            if not params:
                print(f"  WARNING: Empty params for {uuid}, skipping.")
                skipped += 1
                continue

            summary_df = pd.read_csv(files["summary"])
            if summary_df.empty:
                print(f"  WARNING: Empty summary for {uuid}, skipping.")
                skipped += 1
                continue

            for _, srow in summary_df.iterrows():
                row = {**params, **srow.to_dict()}
                all_rows.append(row)

        except Exception as e:
            print(f"  ERROR processing {uuid}: {e}")
            skipped += 1

    if not all_rows:
        print("No rows collected. Exiting.")
        return

    result_df = pd.DataFrame(all_rows)

    # Reorder columns: explicit order first, then any extra columns alphabetically
    present_ordered = [c for c in COLUMN_ORDER if c in result_df.columns]
    extra_cols = sorted(set(result_df.columns) - set(COLUMN_ORDER))
    final_cols = present_ordered + extra_cols

    if extra_cols:
        print(f"  Extra columns found (appended alphabetically): {extra_cols}")

    result_df = result_df[final_cols]

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(result_df)} rows × {len(final_cols)} columns to {OUTPUT_PATH}")
    if skipped:
        print(f"Skipped {skipped} pair(s) due to errors.")
    print("Done.")


if __name__ == "__main__":
    main()