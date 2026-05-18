#!/usr/bin/env bash
# =============================================================================
# generate_all.sh
# 
# Orchestrates all LaTeX generation scripts in the correct order:
#   1. generate_tables.py       (latex_table_group_*.tex)
#   2. generate_appendix.py     (subsections/appendix_*.tex)
#   3. compare_vs_conventional.py (comparison tables vs original DEA)
#   4. compare_vs_pca.py         (comparison tables UMAP vs PCA)
#
# Usage:
#   ./scripts/latex/generate_all.sh            # Full generation
#   ./scripts/latex/generate_all.sh --clean    # Delete old .tex files first
#   ./scripts/latex/generate_all.sh -c         # Same as --clean
#   ./scripts/latex/generate_all.sh --skip-tables  # Skip tables + appendix
#   ./scripts/latex/generate_all.sh -s         # Same as --skip-tables
#   ./scripts/latex/generate_all.sh -c -s      # Clean + skip tables
#
# Flags can be combined (e.g. -cs).
# =============================================================================

set -euo pipefail

# Resolve the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEX_DIR="$PROJECT_ROOT/tex"

# Flags
CLEAN=false
SKIP_TABLES=false

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --clean|-c)
            CLEAN=true
            ;;
        --skip-tables|-s)
            SKIP_TABLES=true
            ;;
        -cs|-sc)
            CLEAN=true
            SKIP_TABLES=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--clean|-c] [--skip-tables|-s]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper: run a Python script and abort on failure
# ---------------------------------------------------------------------------
run_script() {
    local description="$1"
    local script_name="$2"
    shift 2

    echo ""
    echo "============================================="
    echo "  $description"
    echo "  Running: $script_name $*"
    echo "============================================="
    echo ""

    python "$SCRIPT_DIR/$script_name" "$@"

    echo ""
    echo "  ✅ $description — done."
}

# ---------------------------------------------------------------------------
# Clean old .tex files
# ---------------------------------------------------------------------------
if $CLEAN; then
    echo ""
    echo "============================================="
    echo "  Cleaning old .tex files from $TEX_DIR"
    echo "============================================="
    echo ""

    if [ -d "$TEX_DIR" ]; then
        DELETED_COUNT=$(find "$TEX_DIR" -type f -name '*.tex' | wc -l | tr -d ' ')
        if [ "$DELETED_COUNT" -gt 0 ]; then
            find "$TEX_DIR" -type f -name '*.tex' -delete
            echo "  🗑  Deleted $DELETED_COUNT .tex file(s)."
        else
            echo "  ℹ️  No .tex files found — nothing to clean."
        fi
    else
        echo "  ℹ️  $TEX_DIR does not exist — nothing to clean."
        mkdir -p "$TEX_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: Generate per-group landscape tables
# ---------------------------------------------------------------------------
if $SKIP_TABLES; then
    echo ""
    echo "============================================="
    echo "  ⏩ Skipping generate_tables.py (--skip-tables)"
    echo "============================================="
else
    run_script "Step 1/4: Generate per-group landscape tables" \
        "generate_tables.py"

    # -----------------------------------------------------------------------
    # Step 2: Generate appendix subsections from table groups
    # -----------------------------------------------------------------------
    run_script "Step 2/4: Generate appendix subsections" \
        "generate_appendix.py"
fi

# ---------------------------------------------------------------------------
# Step 3: Comparison vs conventional DEA (no dim. reduction)
# ---------------------------------------------------------------------------
run_script "Step 3/4: Comparison vs conventional DEA" \
    "compare_vs_conventional.py"

# ---------------------------------------------------------------------------
# Step 4: Comparison UMAP-DEA vs PCA-DEA
# ---------------------------------------------------------------------------
run_script "Step 4/4: Comparison UMAP-DEA vs PCA-DEA" \
    "compare_vs_pca.py"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================="
echo "  📊 Generation Summary"
echo "============================================="
echo ""

if [ -d "$TEX_DIR" ]; then
    TOTAL_FILES=$(find "$TEX_DIR" -type f -name '*.tex' | wc -l | tr -d ' ')
    echo "  Total .tex files: $TOTAL_FILES"
    echo ""

    # Per-gamma-directory breakdown
    for gamma_dir in "$TEX_DIR"/gamma_*/; do
        [ -d "$gamma_dir" ] || continue
        dir_name="$(basename "$gamma_dir")"
        count=$(find "$gamma_dir" -type f -name '*.tex' | wc -l | tr -d ' ')
        echo "    $dir_name: $count file(s)"

        # Per-type breakdown within each gamma directory
        tables_count=$(find "$gamma_dir"/pca_dea -type f -name 'latex_table_group_*.tex' 2>/dev/null | wc -l | tr -d ' ')
        umap_count=$(find "$gamma_dir"/umap_dea -type f -name 'latex_table_group_*.tex' 2>/dev/null | wc -l | tr -d ' ')
        subs_count=$(find "$gamma_dir"/subsections -type f -name '*.tex' 2>/dev/null | wc -l | tr -d ' ')
        comp_count=$(find "$gamma_dir"/comparison -type f -name '*.tex' 2>/dev/null | wc -l | tr -d ' ')

        [ "$tables_count" -gt 0 ] && echo "      PCA table groups: $tables_count"
        [ "$umap_count" -gt 0 ] && echo "      UMAP table groups: $umap_count"
        [ "$subs_count" -gt 0 ] && echo "      Appendix subsections: $subs_count"
        [ "$comp_count" -gt 0 ] && echo "      Comparison tables: $comp_count"
    done
else
    echo "  No tex/ directory found."
fi

echo ""
echo "============================================="
echo "  🎉 All done."
echo "============================================="