#!/usr/bin/env bash
# Sweep over learning rates: 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
# Varies training.lr while keeping all other hyperparameters at their defaults.

set -euo pipefail

cd "$(dirname "$0")/.."

uv run train.py --multirun \
    training.lr=1e-4,5e-4,1e-3,5e-3,1e-2
