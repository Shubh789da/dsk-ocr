#!/bin/bash
set -e

echo "Container started. Python version:"
python3 --version

echo "Starting RunPod Handler..."
# Run the handler. If it crashes, catch the error and sleep
# This prevents the "Restart Loop" in RunPod so you can read logs
if ! python3 -u handler.py; then
    echo "ERROR: Handler crashed!"
    echo "Sleeping infinity for debugging..."
    sleep infinity
fi
