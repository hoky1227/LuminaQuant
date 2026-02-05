#!/bin/bash

# Production Startup Script for macOS/Linux
# Handles virtual environment activation and auto-restart on crash.

echo "---------------------------------------------------"
echo "[PRODUCTION] Starting Quants Agent in Resilient Mode"
echo "---------------------------------------------------"

# Ensure log directory exists
mkdir -p logs

while true; do
    echo "[INFO] Activating Virtual Environment..."
    # Check for venv in standard locations
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "[WARNING] Virtual environment not found! Attempting to run with system python..."
    fi

    echo "[INFO] Launching Live Trader at $(date)..."
    python3 run_live.py >> logs/crash.log 2>&1
    EXIT_CODE=$?

    echo "[WARNING] Bot crashed or stopped! Exit Code: $EXIT_CODE"
    echo "[INFO] Restarting in 5 seconds... (Press Ctrl+C to stop)"
    sleep 5
done
