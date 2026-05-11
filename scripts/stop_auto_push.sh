#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Removing auto-push cron job..."
crontab -r 2>/dev/null || echo "No crontab found — nothing to remove."

echo ""
echo "Auto-push has been stopped."
echo ""
echo "To clean up leftover files, run:"
echo "  rm $SCRIPT_DIR/auto_push.sh $SCRIPT_DIR/auto_push.log"