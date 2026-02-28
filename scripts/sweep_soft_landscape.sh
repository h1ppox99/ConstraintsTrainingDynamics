#!/usr/bin/env bash
# Soft-penalty landscape sweep
#
# Trains ONLY the soft-penalty model across a grid of:
#   - batch sizes         : 32, 64, 128
#   - learning rates      : 1e-4, 1e-3, 1e-2
#   - penalty weights (λ) : 1.0, 10.0, 100.0
#   - backbone types      : mlp, transformer
#
# That gives 3 × 3 × 3 × 2 = 54 independent runs.
# The loss landscape is plotted every 20 epochs (and at the final epoch).
#
# Each run saves its landscape snapshots under:
#   results/<timestamp>/landscape_soft/epoch_<N>/
#
# Usage:
#   bash scripts/sweep_soft_landscape.sh
#   bash scripts/sweep_soft_landscape.sh 2>&1 | tee logs/sweep_soft_landscape.log

set -euo pipefail

cd "$(dirname "$0")/.."

uv run train.py --multirun \
    models='[soft]' \
    training.epochs=100 \
    training.batch_size=32,64,128 \
    training.lr=1e-4,1e-3,1e-2 \
    training.penalty_weight=1.0,10.0,100.0 \
    model.backbone_type=mlp,transformer \
    landscape.enabled=true \
    landscape.every_n_epochs=20 \
    landscape.grid_size=21 \
    landscape.coord_range=0.3 \
    landscape.log_scale=true \
    landscape.clip_percentile=95.0 \
    wandb.enabled=false
