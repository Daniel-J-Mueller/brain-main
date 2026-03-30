#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="$(basename $0 .sh | sed 's/^run_//')"
# RAS uses GPU 2 alongside hormone control
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$REPO_ROOT" python -m brain.src.$MODULE "$@"
