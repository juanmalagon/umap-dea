"""
Organize /results/ into a clean structure.

Steps:
  1. Recursively discover all CSV files across all subdirectories under /results/.
  2. Group files by UUID, deduplicating (files with same UUID are identical copies).
  3. Read each params_dict CSV to determine pca, N, and gamma.
  4. Move files to the target structure:
       results/gamma_XXX/pca_dea/N_XXX/     (if pca=True)
       results/gamma_XXX/umap_dea/N_XXX/    (if pca=False)
  5. Verify the new structure has all data.
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
    """Format gamma value for directory name.
    e.g. '0.5' -> '0p5', '1.0' -> '1', '2.0' -> '2'.
    """
    f = float(gamma)
    if f == int(f):
        return str(int(f))
    return gamma.replace(".", "p")


def read_params(params_path: Path) -> tuple[str, str, str] | None:
    """Return (pca_value, N, gamma) or None."""
    try:
        text = params_path.read_text()
        lines = [l.strip() for l in text.strip().splitlines()]
        if len(lines) < 2:
            print(f"  WARNING: {params_path.name} has no data row. Skipping.")
            return None
        header = [h.strip() for h in lines[0].split(",")]
        values = [v.strip() for v in lines[1].split(",")]
        row = dict(zip(header, values))
        return row.get("pca", ""), row.get("N", ""), row.get("gamma", "")
    except Exception as e:
        print(f"  ERROR reading {params_path.name}: {e}")
        return None


def dest_dir_for(pca: str, N: str, gamma: str = "") -> Path:
    """Determine the destination directory based on params."""
    gamma_part = f"gamma_{gamma_to_str(gamma)}" if gamma else ""
    n_padded = f"N_{int(N):03d}" if N else f"N_{N}"
    if pca.lower() == "true":
        return RESULTS_DIR / gamma_part / "pca_dea" / n_padded if gamma_part else RESULTS_DIR / "pca_dea" / n_padded
    else:
        return RESULTS_DIR / gamma_part / "umap_dea" / n_padded if gamma_part else RESULTS_DIR / "umap_dea" / n_padded


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
            if file_hash(existing) != file_hash(fpath):
                print(f"  WARNING: {fpath.name} differs from {existing.name} (same UUID). Keeping first, deleting duplicate.")
            else:
                print(f"  Deleting duplicate: {fpath}")
            try:
                fpath.unlink()
            except Exception as e:
                print(f"  ERROR deleting duplicate {fpath}: {e}")
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


# ── step 3 & 4: read params, move to destination ────────────────────

def move_to_structure(uuid_files: dict[str, list[Path]]) -> tuple[int, int, int]:
    """
    Move files to target structure. Returns (moved, skipped, errors).
    """
    moved = 0
    deleted_dup = 0
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

        pca, N, gamma = pd
        dest_dir = dest_dir_for(pca, N, gamma)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src_path in files:
            dest_path = dest_dir / src_path.name

            # Skip if source is already at the correct destination
            if src_path.resolve() == dest_path.resolve():
                continue

            if dest_path.exists():
                if file_hash(src_path) == file_hash(dest_path):
                    # Identical file already at destination — delete the source duplicate
                    print(f"  Deleting duplicate (already at destination): {src_path}")
                    try:
                        src_path.unlink()
                        deleted_dup += 1
                    except Exception as e:
                        print(f"  ERROR deleting {src_path}: {e}")
                        errors += 1
                    continue
                else:
                    print(f"  OVERWRITING (content differs): {dest_path}")
                    dest_path.unlink()
            try:
                shutil.move(str(src_path), str(dest_path))
                moved += 1
            except Exception as e:
                print(f"  ERROR moving {src_path.name}: {e}")
                errors += 1

    return moved, deleted_dup, errors


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
        pca, N, gamma = pd
        dd = dest_dir_for(pca, N, gamma)
        for src in files:
            dest = dd / src.name
            if not dest.exists():
                print(f"  MISSING: {dest}")
                ok = False
    return ok


# ── step 6: cleanup stray files and empty directories ────────────────

VALID_DIR_PATTERN = re.compile(
    r"results/gamma_[^/]+/(?:pca_dea|umap_dea(?:/N_\d+)?)/"  # intentionally left off for Path matching
)


def is_inside_target_structure(file_path: Path) -> bool:
    """Check if a file is inside a valid target directory under results/."""
    try:
        rel = str(file_path.relative_to(RESULTS_DIR))
    except ValueError:
        return False  # Not under RESULTS_DIR

    parts = Path(rel).parts
    if len(parts) < 2:
        return False

    # Must start with gamma_XXX
    if not parts[0].startswith("gamma_"):
        return False

    # Second level: pca_dea or umap_dea
    if parts[1] not in ("pca_dea", "umap_dea"):
        return False

    # Third level must be N_XXX
    if len(parts) < 3 or not parts[2].startswith("N_"):
        return False

    return True


def cleanup_strays(root: Path) -> tuple[int, int]:
    """Delete CSV files outside the target structure and prune empty directories.
    Returns (files_deleted, dirs_deleted).
    """
    files_deleted = 0

    # Delete stray CSV files
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        dp = Path(dirpath)
        for fn in filenames:
            fp = dp / fn
            if fn.endswith(".csv") and not is_inside_target_structure(fp):
                print(f"  Deleting stray file: {fp.relative_to(RESULTS_DIR)}")
                fp.unlink()
                files_deleted += 1

    # Prune empty directories (bottom-up)
    dirs_deleted = 0
    for dirpath, dirnames, _filenames in os.walk(root, topdown=False):
        dp = Path(dirpath)
        if dp == root:
            continue
        try:
            # Only remove if empty (no files, no subdirs)
            if not any(dp.iterdir()):
                dp.rmdir()
                dirs_deleted += 1
        except OSError:
            pass

    return files_deleted, dirs_deleted


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

    print("\nStep 3 & 4: Reading params and moving files to clean structure…")
    moved, deleted_dup, errors = move_to_structure(uuid_groups)
    print(f"  Moved: {moved}, Duplicates deleted: {deleted_dup}, Errors: {errors}")

    print("\nStep 5: Verifying new structure…")
    if verify_structure(uuid_groups):
        print("  All files verified in the new structure ✓")
    else:
        print("  Some files missing in the new structure ✗ — aborting cleanup.")
        return

    print("\nStep 6: Cleaning up stray files and empty directories…")
    stray_files, stray_dirs = cleanup_strays(RESULTS_DIR)
    print(f"  Stray files deleted: {stray_files}, Empty directories removed: {stray_dirs}")

    print("\nDone. Final structure:")
    for dirpath, dirnames, filenames in os.walk(RESULTS_DIR):
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