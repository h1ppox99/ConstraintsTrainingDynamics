#!/usr/bin/env bash
# Sweep over batch sizes: 16, 32, 64, 128, 256
# Varies training.batch_size while keeping all other hyperparameters at their defaults.

set -euo pipefail

cd "$(dirname "$0")/.."

uv run train.py --multirun \
    training.batch_size=16,32,64,128,256
