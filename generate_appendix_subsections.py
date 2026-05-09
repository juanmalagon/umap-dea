#!/usr/bin/env python3
r"""
Generate appendix subsection .tex files from latex_table_group_*.tex files.

Each latex_table_group file is classified along two dimensions:
  - Method: "PCA-DEA" or "UMAP-DEA" (parsed from the \caption line)
  - RTS type: "CRS" or "VRS" (parsed from the comment header)

Files are then grouped into 4 buckets: pca_crs, pca_vrs, umap_crs, umap_vrs.
For each non-empty group, a subsection .tex file is generated in tex/subsections/,
with \input{} directives for the constituent tables, sorted by N (ascending).
A comment at the top lists the source files for traceability, and the filename
includes a short hash of the sorted source filenames.
"""

import hashlib
import os
import re

# Paths
TEX_DIR = "tex"
SUBSECTIONS_DIR = os.path.join(TEX_DIR, "subsections")

# Regex patterns
HEADER_RTS_RE = re.compile(r"^%\s+(CRS|VRS)[,\s]")
CAPTION_METHOD_RE = re.compile(r"\\caption\{.*(PCA-DEA|UMAP-DEA)")
N_RE = re.compile(r"N=(\d+)")


def parse_table_group_file(filepath: str) -> dict | None:
    """
    Parse a latex_table_group_*.tex file to extract:
      - method: "pca" or "umap"
      - rts: "crs" or "vrs"
      - N: integer (number of inputs)
    Returns None if parsing fails.
    """
    result = {"method": None, "rts": None, "N": None}

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse method from \caption{} (PCA-DEA or UMAP-DEA)
    caption_match = CAPTION_METHOD_RE.search(content)
    if caption_match:
        method_str = caption_match.group(1)
        result["method"] = "pca" if "PCA" in method_str else "umap"
    else:
        # Fallback: check filename for PCA/umap
        basename = os.path.basename(filepath)
        if "_PCA_" in basename:
            result["method"] = "pca"
        elif "_umap_" in basename.lower():
            result["method"] = "umap"
        else:
            print(f"  Warning: cannot determine method for {filepath}")
            return None

    # Parse RTS type from % CRS or % VRS comment header
    for line in content.splitlines():
        header_match = HEADER_RTS_RE.match(line)
        if header_match:
            rts_str = header_match.group(1).lower()
            result["rts"] = rts_str
            break

    if result["rts"] is None:
        # Fallback: parse from filename
        basename = os.path.basename(filepath)
        if "_crs_" in basename:
            result["rts"] = "crs"
        elif "_vrs_" in basename:
            result["rts"] = "vrs"
        else:
            print(f"  Warning: cannot determine RTS type for {filepath}")
            return None

    # Parse N from the comment header or caption
    n_match = N_RE.search(content)
    if n_match:
        result["N"] = int(n_match.group(1))
    else:
        # Fallback: parse from filename
        basename = os.path.basename(filepath)
        n_match = N_RE.search(basename)
        if n_match:
            result["N"] = int(n_match.group(1))
        else:
            print(f"  Warning: cannot determine N for {filepath}")
            return None

    return result


def generate_subsection(method_key: str, rts_key: str, files: list[str]) -> str:
    """
    Generate the LaTeX content for one subsection file.

    Parameters:
      method_key: "pca" or "umap"
      rts_key: "crs" or "vrs"
      files: sorted list of (N, filename) tuples, already sorted by N
    """
    method_display = "PCA-DEA" if method_key == "pca" else "UMAP-DEA"
    rts_display = rts_key.upper()

    # Build the subsection content
    lines = []
    lines.append("% ============================================================")
    lines.append(f"% Generated from:")
    for _, fname in files:
        lines.append(f"%   - {fname}")
    # Compute hash
    sorted_fnames = [fname for _, fname in files]
    hash_input = "\n".join(sorted_fnames).encode("utf-8")
    short_hash = hashlib.sha256(hash_input).hexdigest()[:8]
    lines.append(f"% Hash: {short_hash}")
    lines.append("% ============================================================")
    lines.append("")
    lines.append(f"\\subsection{{{method_display} {rts_display}}}\\label{{apd:{method_key}_{rts_key}}}")
    lines.append("")

    for _, fname in files:
        filepath = os.path.join(TEX_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        lines.append(content)
        lines.append("")

    return "\n".join(lines), short_hash


def main():
    os.makedirs(SUBSECTIONS_DIR, exist_ok=True)

    # Step 1: collect all latex_table_group files
    all_files = []
    for fname in os.listdir(TEX_DIR):
        if fname.startswith("latex_table_group_") and fname.endswith(".tex"):
            all_files.append(fname)

    if not all_files:
        print("No latex_table_group_*.tex files found.")
        return

    print(f"Found {len(all_files)} table group files.")

    # Step 2: parse each file
    buckets: dict[tuple[str, str], list[tuple[int, str]]] = {
        ("pca", "crs"): [],
        ("pca", "vrs"): [],
        ("umap", "crs"): [],
        ("umap", "vrs"): [],
    }

    for fname in sorted(all_files):
        filepath = os.path.join(TEX_DIR, fname)
        info = parse_table_group_file(filepath)
        if info is None:
            continue
        key = (info["method"], info["rts"])
        if key not in buckets:
            print(f"  Warning: unexpected key {key} for {fname}, skipping")
            continue
        buckets[key].append((info["N"], fname))
        print(f"  {fname}: method={info['method']}, rts={info['rts']}, N={info['N']}")

    # Step 3: sort each bucket by N (ascending)
    for key in buckets:
        buckets[key].sort(key=lambda x: x[0])

    # Step 4: generate subsection files
    generated = []
    for (method_key, rts_key), files in buckets.items():
        if not files:
            continue

        content, short_hash = generate_subsection(method_key, rts_key, files)
        out_fname = f"appendix_{method_key}_{rts_key}_{short_hash}.tex"
        out_path = os.path.join(SUBSECTIONS_DIR, out_fname)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        generated.append(out_fname)
        print(f"\nCreated: {out_path}")

    print(f"\nDone. Generated {len(generated)} subsection file(s) in {SUBSECTIONS_DIR}/")


if __name__ == "__main__":
    main()