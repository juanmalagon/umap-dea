#!/bin/bash
set -e

REPO_DIR="/home/totoi/repos/umap-dea"
BRANCH="feat/export_results"
LOG_FILE="$REPO_DIR/scripts/auto_push.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cd "$REPO_DIR"

log "=== Auto-push check started ==="

# Fetch latest remote state for this branch
log "Fetching origin/$BRANCH..."
git fetch origin "$BRANCH" 2>&1 | tee -a "$LOG_FILE"

# Count how many commits local is ahead of remote
AHEAD_COUNT=$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo "0")

if [ "$AHEAD_COUNT" -gt 0 ]; then
    log "Found $AHEAD_COUNT unpushed commit(s). Pushing..."
    git push origin "$BRANCH" 2>&1 | tee -a "$LOG_FILE"
    log "Push completed successfully."
else
    log "No unpushed commits. Nothing to do."
fi

log "=== Auto-push check finished ==="