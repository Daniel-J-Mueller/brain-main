#!/bin/bash
# Broca's area runs on GPU 3 with the motor cortex
CUDA_VISIBLE_DEVICES=3 python -m brain.src.language_areas.brocas_area "$@"
