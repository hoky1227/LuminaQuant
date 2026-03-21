#!/bin/bash

set -euo pipefail

echo "---------------------------------------------------"
echo "[PRODUCTION] Starting LuminaQuant in Resilient Mode"
echo "---------------------------------------------------"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs .omx/tmp

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is required but not found in PATH."
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash run_bot.sh [start_live_session.sh options]

Resilient wrapper around `uv run lq live` via scripts/ops/start_live_session.sh.
- First launch runs the full controlled startup flow
- Crash restarts skip heavy prep steps
- Graceful stop / clean exit does not restart

Common examples:
  bash run_bot.sh --dsn 'postgresql:///luminaquant'
  bash run_bot.sh --real --allow-real --dsn 'postgresql:///luminaquant'
  bash run_bot.sh --transport ws --dsn 'postgresql:///luminaquant'
EOF
    echo
    bash scripts/ops/start_live_session.sh --help
    exit 0
fi

FIRST_ATTEMPT=1

while true; do
    LAUNCH_MARKER=".omx/tmp/run_bot_live_started.marker"
    rm -f "$LAUNCH_MARKER"

    CMD=(bash scripts/ops/start_live_session.sh --launch-marker "$LAUNCH_MARKER")
    if [[ "$FIRST_ATTEMPT" != "1" ]]; then
        CMD+=(--skip-init-schema --skip-refresh --skip-validate --skip-preflight)
    fi
    if [[ "$#" -gt 0 ]]; then
        CMD+=("$@")
    fi

    echo "[INFO] Launching controlled live session at $(date)..."
    printf '[INFO] Command: '
    printf '%q ' "${CMD[@]}"
    printf '\n'

    set +e
    "${CMD[@]}" >> logs/crash.log 2>&1
    EXIT_CODE=$?
    set -e

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo "[INFO] Live session exited cleanly. Not restarting."
        exit 0
    fi

    if [[ "$EXIT_CODE" -eq 130 || "$EXIT_CODE" -eq 143 ]]; then
        echo "[INFO] Live session interrupted. Not restarting."
        exit "$EXIT_CODE"
    fi

    if [[ ! -f "$LAUNCH_MARKER" ]]; then
        echo "[ERROR] Live session failed before launch/preflight completed. Check logs/crash.log."
        exit "$EXIT_CODE"
    fi

    FIRST_ATTEMPT=0
    echo "[WARNING] Live session crashed after launch. Exit code: $EXIT_CODE"
    echo "[INFO] Restarting in 5 seconds... (Use stop-file or Ctrl+C to stop)"
    sleep 5
done
