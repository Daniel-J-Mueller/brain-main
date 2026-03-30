#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODULE="$(basename $0 .sh | sed 's/^run_//')"
# Thalamus runs alongside the DMN on GPU 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO_ROOT" python -m brain.src.$MODULE "$@"
