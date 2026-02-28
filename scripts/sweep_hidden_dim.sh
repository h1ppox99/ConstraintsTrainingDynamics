#!/usr/bin/env bash
# Sweep over MLP hidden layer dimensions: 32, 64, 128, 256
# Varies model.hidden_dim while keeping all other hyperparameters at their defaults.

set -euo pipefail

cd "$(dirname "$0")/.."

uv run train.py --multirun \
    model.hidden_dim=32,64,128,256
