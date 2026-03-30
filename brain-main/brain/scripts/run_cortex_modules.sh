#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="$(basename $0 .sh | sed 's/^run_//')"
# Context and salience cortex run on GPU 2
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$REPO_ROOT" python -m brain.src.$MODULE "$@"
