"""
Training dynamics metrics for constraint-handling experiments.

Exposes individual metric functions and the MetricsTracker for use
in training loops.
"""

from .metrics import (
    compute_loss,
    compute_constraint_violations,
    compute_grad_cosine_similarity,
    compute_grad_snr,
    compute_hessian_top_eigenvalues,
    compute_jacobian_effective_rank,
    MetricsTracker,
)

__all__ = [
    "compute_loss",
    "compute_constraint_violations",
    "compute_grad_cosine_similarity",
    "compute_grad_snr",
    "compute_hessian_top_eigenvalues",
    "compute_jacobian_effective_rank",
    "MetricsTracker",
]
