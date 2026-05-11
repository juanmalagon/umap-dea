"""
Organize /results/ into a clean structure.

Steps:
  1. Recursively discover all CSV files across all subdirectories under /results/.
  2. Group files by UUID, deduplicating (files with same UUID are identical copies).
  3. Read each params_dict CSV to determine pca, umap_n_neighbors, and gamma.
  4. Copy files to the target structure:
       results/gamma_XXX/pca_dea/           (if pca=True)
       results/gamma_XXX/umap_dea/k_XX/     (if pca=False)
  5. Verify the new structure has all data.
  6. Delete all old /rubbish/ folders (simulation_results_*).
"""

import os
import re
import shutil
import hashlib
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FILE_TYPES = ["evaluation_df", "summary_df", "errors_list", "params_dict"]

# ── helpers ──────────────────────────────────────────────────────────

UUID_RE = re.compile(
    rf"(?:{'|'.join(FILE_TYPES)})_([a-f0-9-]+)\.csv$", re.IGNORECASE
)


def extract_uuid(filename: str) -> str | None:
    m = UUID_RE.search(filename)
    return m.group(1) if m else None


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def gamma_to_str(gamma: str) -> str:
    """Format gamma value for directory name, e.g. '0.5' -> '0p5'."""
    return gamma.replace(".", "p")


def read_params(params_path: Path) -> tuple[str, str, str] | None:
    """Return (pca_value, umap_n_neighbors, gamma) or None."""
    try:
        text = params_path.read_text()
        lines = [l.strip() for l in text.strip().splitlines()]
        if len(lines) < 2:
            print(f"  WARNING: {params_path.name} has no data row. Skipping.")
            return None
        header = [h.strip() for h in lines[0].split(",")]
        values = [v.strip() for v in lines[1].split(",")]
        row = dict(zip(header, values))
        return row.get("pca", ""), row.get("umap_n_neighbors", ""), row.get("gamma", "")
    except Exception as e:
        print(f"  ERROR reading {params_path.name}: {e}")
        return None


def dest_dir_for(pca: str, k: str, gamma: str = "") -> Path:
    """Determine the destination directory based on params."""
    gamma_part = f"gamma_{gamma_to_str(gamma)}" if gamma else ""
    if pca.lower() == "true":
        return RESULTS_DIR / gamma_part / "pca_dea" if gamma_part else RESULTS_DIR / "pca_dea"
    else:
        k_padded = f"k_{int(k):02d}" if k else f"k_{k}"
        return RESULTS_DIR / gamma_part / "umap_dea" / k_padded if gamma_part else RESULTS_DIR / "umap_dea" / k_padded


# ── step 1: discover all CSV files recursively ──────────────────────

def discover_all_csv_files(root: Path) -> list[Path]:
    """Walk *all* subdirectories (recursively) and return CSV file paths."""
    csv_files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".csv"):
                csv_files.append(Path(dirpath) / fn)
    return csv_files


# ── step 2: group by UUID, deduplicate ──────────────────────────────

def group_and_deduplicate(csv_files: list[Path]) -> dict[str, list[Path]]:
    """
    Return a dict uuid -> [list of file paths].
    Only one copy per UUID is kept (the first encountered that has a params_dict).
    """
    uuid_candidates: dict[str, dict[str, Path]] = {}  # uuid -> {filetype: path}

    for fpath in csv_files:
        uuid = extract_uuid(fpath.name)
        if not uuid:
            continue

        # Determine file type prefix
        filetype = None
        for ft in FILE_TYPES:
            if fpath.name.startswith(ft + "_"):
                filetype = ft
                break
        if filetype is None:
            continue

        if uuid not in uuid_candidates:
            uuid_candidates[uuid] = {}

        existing = uuid_candidates[uuid].get(filetype)
        if existing is not None and existing != fpath:
            # Same UUID and filetype found in multiple places — keep whichever we saw first
            # (they should be identical; if not, log a warning)
            if file_hash(existing) != file_hash(fpath):
                print(f"  WARNING: {fpath.name} differs from {existing.name} (same UUID). Keeping first.")
        else:
            uuid_candidates[uuid][filetype] = fpath

    # Convert to uuid -> list[Path], ensure params_dict exists
    result: dict[str, list[Path]] = {}
    orphaned: list[Path] = []
    for uuid, ft_map in uuid_candidates.items():
        if "params_dict" not in ft_map:
            for p in ft_map.values():
                orphaned.append(p)
            continue
        result[uuid] = list(ft_map.values())

    if orphaned:
        print(f"\nFound {len(orphaned)} files with no params_dict (orphaned). Listing:")
        for p in sorted(orphaned):
            print(f"  {p}")

    return result


# ── step 3 & 4: read params, copy to destination ────────────────────

def copy_to_structure(uuid_files: dict[str, list[Path]]) -> tuple[int, int, int]:
    """
    Copy files to target structure. Returns (copied, skipped, errors).
    """
    copied = 0
    skipped = 0
    errors = 0

    for uuid, files in sorted(uuid_files.items()):
        params_path = None
        for f in files:
            if f.name.startswith("params_dict_"):
                params_path = f
                break

        if params_path is None:
            print(f"  WARNING: No params_dict for UUID {uuid}. Skipping {len(files)} files.")
            errors += len(files)
            continue

        pd = read_params(params_path)
        if pd is None:
            print(f"  WARNING: Could not parse params for UUID {uuid}. Skipping {len(files)} files.")
            errors += len(files)
            continue

        pca, k, gamma = pd
        dest_dir = dest_dir_for(pca, k, gamma)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src_path in files:
            dest_path = dest_dir / src_path.name
            if dest_path.exists():
                if file_hash(src_path) == file_hash(dest_path):
                    skipped += 1
                    continue
                else:
                    print(f"  OVERWRITING (content differs): {dest_path}")
                    dest_path.unlink()
            try:
                shutil.copy2(str(src_path), str(dest_path))
                copied += 1
            except Exception as e:
                print(f"  ERROR copying {src_path.name}: {e}")
                errors += 1

    return copied, skipped, errors


# ── step 5: verify ──────────────────────────────────────────────────

def verify_structure(original_groups: dict[str, list[Path]]) -> bool:
    """Verify all files exist in the new structure."""
    ok = True
    for uuid, files in original_groups.items():
        params_path = None
        for f in files:
            if f.name.startswith("params_dict_"):
                params_path = f
                break
        if params_path is None:
            continue
        pd = read_params(params_path)
        if pd is None:
            continue
        pca, k, gamma = pd
        dd = dest_dir_for(pca, k, gamma)
        for src in files:
            dest = dd / src.name
            if not dest.exists():
                print(f"  MISSING: {dest}")
                ok = False
    return ok


# ── step 6: delete old folders ──────────────────────────────────────

RUBBISH_PATTERNS = re.compile(
    r"^simulation_results_",
    re.IGNORECASE,
)


def delete_rubbish_folders(root: Path) -> list[Path]:
    """Delete all top-level folders matching the rubbish pattern, return their paths."""
    deleted: list[Path] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and RUBBISH_PATTERNS.match(entry.name):
            print(f"  Deleting: {entry}")
            shutil.rmtree(entry)
            deleted.append(entry)
    return deleted


# ── main ────────────────────────────────────────────────────────────

def main():
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    print("=" * 60)
    print("Step 1: Recursively discovering all CSV files…")
    all_csv = discover_all_csv_files(RESULTS_DIR)
    print(f"  Found {len(all_csv)} CSV files total.")

    print("\nStep 2: Grouping by UUID and deduplicating…")
    uuid_groups = group_and_deduplicate(all_csv)
    print(f"  Found {len(uuid_groups)} unique simulation runs with params_dict.")

    unique_csv_count = sum(len(v) for v in uuid_groups.values())
    print(f"  Total CSV files (after dedup): {unique_csv_count}")

    print("\nStep 3 & 4: Reading params and copying files to clean structure…")
    copied, skipped, errors = copy_to_structure(uuid_groups)
    print(f"  Copied: {copied}, Skipped (already present): {skipped}, Errors: {errors}")

    print("\nStep 5: Verifying new structure…")
    if verify_structure(uuid_groups):
        print("  All files verified in the new structure ✓")
    else:
        print("  Some files missing in the new structure ✗ — aborting cleanup.")
        return

    print("\nStep 6: Deleting old rubbish folders…")
    deleted = delete_rubbish_folders(RESULTS_DIR)
    print(f"  Deleted {len(deleted)} folders: {[d.name for d in deleted]}")

    print("\nDone. Final structure:")
    for dirpath, dirnames, filenames in os.walk(RESULTS_DIR):
        if any(RUBBISH_PATTERNS.match(Path(dirpath).name) for _ in [1]):
            continue
        level = Path(dirpath).relative_to(RESULTS_DIR).parts
        indent = "  " * (len(level))
        print(f"{indent}{Path(dirpath).name}/")
        subindent = "  " * (len(level) + 1)
        for fn in sorted(filenames):
            print(f"{subindent}{fn}")

    total_new = sum(
        1 for dirpath, _dirnames, filenames in os.walk(RESULTS_DIR) for fn in filenames
    )
    print(f"\nTotal files in clean structure: {total_new}")


if __name__ == "__main__":
    main()