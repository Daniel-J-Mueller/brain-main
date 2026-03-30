#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="$(basename $0 .sh | sed 's/^run_//')"
# Trainer updates run on GPU 3
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$REPO_ROOT" python -m brain.src.$MODULE "$@"
