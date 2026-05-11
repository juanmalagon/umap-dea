"""
Sort results in /results/ into subfolders based on configuration parameters.

- Splits into /results/pca_dea/ (pca=True) and /results/umap_dea/ (pca=False)
- Further splits umap_dea/ into subfolders by umap_n_neighbors value:
  /results/umap_dea/k_05/, /results/umap_dea/k_15/, etc.
"""

import os
import re
import shutil
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PARAMS_PREFIX = "params_dict_"
CSV_SUFFIX = ".csv"

# File types associated with each UUID run
FILE_TYPES = ["evaluation_df", "summary_df", "errors_list", "params_dict"]


def extract_uuid(filename: str) -> str | None:
    """Extract UUID from a filename like 'params_dict_<uuid>.csv'."""
    match = re.match(rf"(?:{'|'.join(FILE_TYPES)})_([a-f0-9-]+)\.csv", filename, re.IGNORECASE)
    return match.group(1) if match else None


def read_params_file(params_path: Path) -> tuple[str, str] | None:
    """
    Read a params_dict CSV file and return (pca_value, umap_n_neighbors).
    Returns None if the file can't be read or parsed.
    """
    try:
        with open(params_path, "r") as f:
            lines = f.readlines()
        if len(lines) < 2:
            print(f"  WARNING: {params_path.name} has no data row. Skipping.")
            return None

        header = [h.strip() for h in lines[0].strip().split(",")]
        values = [v.strip() for v in lines[1].strip().split(",")]

        # Build dict from header-value pairs
        row = dict(zip(header, values))
        pca = row.get("pca", "")
        k = row.get("umap_n_neighbors", "")
        return pca, k
    except Exception as e:
        print(f"  ERROR reading {params_path.name}: {e}")
        return None


def main():
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    # Collect all CSV files in the results directory (non-recursive)
    all_csv_files = [f for f in RESULTS_DIR.iterdir() if f.is_file() and f.suffix == ".csv"]

    # Group files by UUID
    uuid_files: dict[str, list[Path]] = {}
    for csv_file in all_csv_files:
        uuid = extract_uuid(csv_file.name)
        if uuid:
            uuid_files.setdefault(uuid, []).append(csv_file)

    if not uuid_files:
        print("No result files found to sort.")
        return

    print(f"Found {len(uuid_files)} unique simulation runs to sort.\n")

    moved_count = 0
    skipped_count = 0
    error_count = 0

    for uuid, files in sorted(uuid_files.items()):
        # Find the params_dict file for this UUID
        params_file = None
        for f in files:
            if f.name.startswith(PARAMS_PREFIX):
                params_file = f
                break

        if params_file is None:
            print(f"  WARNING: No params_dict found for UUID {uuid}. Skipping {len(files)} files.")
            error_count += len(files)
            continue

        params_data = read_params_file(params_file)
        if params_data is None:
            print(f"  WARNING: Could not parse params for UUID {uuid}. Skipping.")
            error_count += len(files)
            continue

        pca, k = params_data

        # Determine destination directory
        if pca.lower() == "true":
            dest_dir = RESULTS_DIR / "pca_dea"
        else:
            # Zero-pad k to 2 digits (e.g., 5 -> "05", 15 -> "15")
            k_padded = f"k_{int(k):02d}" if k else f"k_{k}"
            dest_dir = RESULTS_DIR / "umap_dea" / k_padded

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Move all files for this UUID
        for file_path in files:
            dest_path = dest_dir / file_path.name
            try:
                if dest_path.exists():
                    print(f"  OVERWRITING: {dest_path}")
                    os.remove(dest_path)
                shutil.move(str(file_path), str(dest_path))
                moved_count += 1
            except Exception as e:
                print(f"  ERROR moving {file_path.name}: {e}")
                error_count += 1

    print(f"\nDone. Moved {moved_count} files, skipped {skipped_count}, errors {error_count}.")


if __name__ == "__main__":
    main()