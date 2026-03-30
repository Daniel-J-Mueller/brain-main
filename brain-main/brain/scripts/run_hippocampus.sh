#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="$(basename $0 .sh | sed 's/^run_//')"
# Hippocampus shares GPU 1 with the DMN
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO_ROOT" python -m brain.src.$MODULE "$@"
