#!/usr/bin/env bash
# ============================================================================
# Soft-penalty LR search (MLP h128×3)
#
# For each (batch_size, penalty_weight) pair, sweeps learning rate on a fine
# log-spaced grid to find the best training configuration.
#
# Fixed backbone: MLP with hidden_dim=128, n_hidden=3
# Landscapes: disabled (speed)
# Hessian:    disabled (speed)
#
# Grid:
#   - lr             : 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
#   - batch_size     : 32, 64, 128
#   - penalty_weight : 1.0, 10.0, 100.0
#
# Total runs: 7 × 3 × 3 = 63 independent trainings
#
# Results are stored in multirun/<date>/<time>/ by Hydra.
#
# Usage:
#   bash scripts/sweep_soft_lr_search.sh
#   bash scripts/sweep_soft_lr_search.sh 2>&1 | tee logs/sweep_soft_lr_search.log
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

uv run train.py --multirun \
    models='[soft]' \
    training.epochs=150 \
    training.lr=1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2 \
    training.batch_size=32,64,128 \
    training.penalty_weight=1.0,10.0,100.0 \
    model.backbone_type=mlp \
    model.hidden_dim=128 \
    model.n_hidden=3 \
    landscape.enabled=false \
    metrics.log_hessian_every=0 \
    wandb.enabled=true
