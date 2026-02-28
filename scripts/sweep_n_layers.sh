#!/usr/bin/env bash
# Sweep over number of hidden layers: 1, 2, 3
# Varies model.n_hidden while keeping all other hyperparameters at their defaults.

set -euo pipefail

cd "$(dirname "$0")/.."

uv run train.py --multirun \
    model.n_hidden=1,2,3
