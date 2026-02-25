"""
Training dynamics metrics for constraint-handling experiments.

Provides **stateless** functions and a lightweight **MetricsTracker** to log
the following quantities during training:

1.  loss                  – scalar training loss
2.  constraint_violations – per-type (ineq / eq) violation statistics
3.  n_violated            – number of violated constraints
4.  grad_cosine_sim       – cosine similarity between objective and constraint
                            penalty gradients (measures gradient conflict)
5.  grad_snr              – gradient signal-to-noise ratio  (per-layer & global)
6.  hessian_top_eig       – top-k eigenvalues of the loss Hessian
                            (via stochastic power iteration – call periodically)
7.  jacobian_eff_rank     – effective rank of the constraint Jacobian

All functions return plain Python dicts ready for wandb / CSV / JSON logging.

Example
-------
    tracker = MetricsTracker(problem, log_hessian_every=50)

    for step, (X, Y_true) in enumerate(loader):
        Y_pred = model(X)
        loss = ...
        loss.backward()
        metrics = tracker.step(model, X, Y_pred, loss)
        wandb.log(metrics, step=step)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Loss  (trivial – included for completeness / consistent dict key)
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(loss: torch.Tensor) -> dict:
    """Return ``{'loss': float}`` from a scalar loss tensor."""
    return {"loss": loss.detach().item()}


# ═══════════════════════════════════════════════════════════════════════════
#  2 & 3.  Constraint violations
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_constraint_violations(problem, X: torch.Tensor, Y: torch.Tensor,
                                  eps: float = 1e-6) -> dict:
    """
    Constraint-violation statistics.

    Parameters
    ----------
    problem : QCQPProblem
    X       : (b, e)  input parameters
    Y       : (b, n)  predicted solutions
    eps     : threshold below which a residual is considered satisfied

    Returns
    -------
    dict with keys:
        ineq_viol_mean, ineq_viol_max, ineq_viol_sum,
        eq_viol_mean,   eq_viol_max,   eq_viol_sum,
        total_viol_mean, total_viol_max,
        n_ineq_violated, n_eq_violated, n_total_violated,
        frac_ineq_violated, frac_eq_violated
    """
    ineq = problem.get_ineq_res(X, Y)   # (b, m)  – ReLU'd, ≥ 0
    eq   = problem.get_eq_res(X, Y)     # (b, e)  – absolute value, ≥ 0

    # Per-sample violation counts
    n_ineq_viol = (ineq > eps).float().sum()       # scalar
    n_eq_viol   = (eq   > eps).float().sum()

    total_constraints = ineq.numel() + eq.numel()
    n_total = n_ineq_viol + n_eq_viol

    out = {
        # --- magnitudes ---
        "ineq_viol_mean":  ineq.mean().item(),
        "ineq_viol_max":   ineq.max().item(),
        "ineq_viol_sum":   ineq.sum().item(),
        "eq_viol_mean":    eq.mean().item(),
        "eq_viol_max":     eq.max().item(),
        "eq_viol_sum":     eq.sum().item(),
        "total_viol_mean": torch.cat([ineq, eq], dim=1).mean().item(),
        "total_viol_max":  max(ineq.max().item(), eq.max().item()),
        # --- counts ---
        "n_ineq_violated":     int(n_ineq_viol.item()),
        "n_eq_violated":       int(n_eq_viol.item()),
        "n_total_violated":    int(n_total.item()),
        "frac_ineq_violated":  n_ineq_viol.item() / max(ineq.numel(), 1),
        "frac_eq_violated":    n_eq_viol.item()   / max(eq.numel(), 1),
        "frac_total_violated": n_total.item()      / max(total_constraints, 1),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Gradient cosine similarity  (objective ↔ constraint penalty)
# ═══════════════════════════════════════════════════════════════════════════

def compute_grad_cosine_similarity(
    model: nn.Module,
    problem,
    X: torch.Tensor,
    Y: torch.Tensor,
    penalty_weight: float = 1.0,
) -> dict:
    """
    Cosine similarity between the gradient of the objective loss and the
    gradient of the constraint-violation penalty.

    cos(∇_θ L_obj, ∇_θ L_constr)  ∈ [-1, 1]

    A value near **+1** means both losses push parameters in the same
    direction;  **-1** means they are in conflict;  **0** signals
    orthogonality.

    Parameters
    ----------
    model          : the network whose parameters carry gradients
    problem        : QCQPProblem
    X, Y           : current mini-batch  (Y = model output, must retain grad graph)
    penalty_weight : λ used in the penalty loss (for scaling, but cos is
                     scale-invariant so this only matters for numerical stability)

    Returns
    -------
    dict with ``grad_cosine_sim``
    """
    # --- Gradient of objective term ---
    obj_loss = problem.get_objective_loss(Y, X).mean()
    obj_grads = torch.autograd.grad(
        obj_loss, model.parameters(), retain_graph=True, allow_unused=True,
    )
    g_obj = torch.cat([g.flatten() for g in obj_grads if g is not None])

    # --- Gradient of constraint-violation term ---
    resid = problem.get_resid(X, Y)
    viol_loss = (resid ** 2).sum(dim=1).mean()
    viol_grads = torch.autograd.grad(
        viol_loss, model.parameters(), retain_graph=True, allow_unused=True,
    )
    g_viol = torch.cat([g.flatten() for g in viol_grads if g is not None])

    cos = torch.nn.functional.cosine_similarity(
        g_obj.unsqueeze(0), g_viol.unsqueeze(0),
    ).item()

    return {
        "grad_cosine_sim": cos,
        "grad_obj_norm":   g_obj.norm().item(),
        "grad_viol_norm":  g_viol.norm().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5.  Gradient signal-to-noise ratio
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_grad_snr(model: nn.Module) -> dict:
    """
    Gradient signal-to-noise ratio **after** a ``loss.backward()`` call.

    For each parameter group (layer) with gradient **g**:

        SNR_layer = ‖mean(g)‖ / std(g)

    Global SNR is computed on the full flattened gradient vector.

    A *high* SNR means the gradient is a reliable signal; a *low* SNR
    indicates noisy / conflicting gradient directions.

    Returns
    -------
    dict with ``grad_snr_global`` and ``grad_snr/<layer_name>``
    """
    all_grads = []
    layer_snrs = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().flatten()
        all_grads.append(g)

        std = g.std()
        mean_abs = g.mean().abs()
        snr = (mean_abs / (std + 1e-12)).item()
        layer_snrs[f"grad_snr/{name}"] = snr

    if len(all_grads) == 0:
        return {"grad_snr_global": 0.0}

    g_all = torch.cat(all_grads)
    global_snr = (g_all.mean().abs() / (g_all.std() + 1e-12)).item()

    out = {"grad_snr_global": global_snr}
    out.update(layer_snrs)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  6.  Hessian top eigenvalues  (stochastic power iteration)
# ═══════════════════════════════════════════════════════════════════════════

def _hvp(loss_fn, params, vec):
    """
    Hessian-vector product  H · v  using double back-prop.

    Parameters
    ----------
    loss_fn : callable() → scalar  (must create a fresh computation graph)
    params  : list of Parameter
    vec     : 1-D tensor of length sum(p.numel() for p in params)

    Returns
    -------
    Hv : 1-D tensor, same shape as *vec*
    """
    # First-order gradients (create_graph=True so we can differentiate again)
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

    flat_grad = torch.cat([
        (g if g is not None else torch.zeros_like(p)).flatten()
        for g, p in zip(grads, params)
    ])

    # Second-order:  ∂(flat_grad · vec) / ∂params  =  H · vec
    grad_vec_prod = (flat_grad * vec).sum()
    hvp_grads = torch.autograd.grad(grad_vec_prod, params, allow_unused=True)

    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).flatten()
        for g, p in zip(hvp_grads, params)
    ])


def compute_hessian_top_eigenvalues(
    loss_fn,
    model: nn.Module,
    k: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> dict:
    """
    Top-*k* eigenvalues of the loss Hessian via stochastic power iteration
    (Lanczos-like deflation).

    **Expensive** — call every N steps (e.g. 50–100).

    Parameters
    ----------
    loss_fn  : callable() → scalar loss  (will be called multiple times;
               each call must produce a fresh computation graph)
    model    : the model whose named_parameters are used
    k        : number of eigenvalues to return  (default 3)
    max_iter : power-iteration steps per eigenvalue
    tol      : convergence tolerance on eigenvalue change

    Returns
    -------
    dict with ``hessian_eig_1 .. hessian_eig_k`` and ``hessian_eig_max``
    """
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    device = params[0].device
    dtype = params[0].dtype

    eigenvalues: list[float] = []
    eigenvectors: list[torch.Tensor] = []

    for i in range(k):
        # Random initial vector
        v = torch.randn(d, device=device, dtype=dtype)
        v = v / v.norm()

        lam_prev = 0.0
        for _ in range(max_iter):
            Hv = _hvp(loss_fn, params, v)

            # Deflate: project out previously found eigenvectors
            for ev in eigenvectors:
                Hv = Hv - (ev @ Hv) * ev

            lam = (v @ Hv).item()
            v_new = Hv / (Hv.norm() + 1e-12)
            v = v_new

            if abs(lam - lam_prev) < tol * max(abs(lam), 1.0):
                break
            lam_prev = lam

        eigenvalues.append(lam)
        eigenvectors.append(v.detach())

    out = {f"hessian_eig_{i+1}": eigenvalues[i] for i in range(k)}
    out["hessian_eig_max"] = max(eigenvalues)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  7.  Effective rank of the constraint Jacobian
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_jacobian_effective_rank(problem, Y: torch.Tensor) -> dict:
    """
    Effective rank of the constraint Jacobian  dg/dy.

    The effective rank is the exponential of the Shannon entropy of the
    normalised singular values:

        σ̃_i = σ_i / Σ σ_j
        eff_rank = exp( -Σ σ̃_i log σ̃_i )

    A value of 1 means the Jacobian is rank-1; a value equal to
    min(m+e, n) means all singular values are equal (full rank).

    Parameters
    ----------
    problem : QCQPProblem
    Y       : (b, n)  predicted solutions

    Returns
    -------
    dict with ``jacobian_eff_rank`` (batch-averaged),
    ``jacobian_rank_ratio`` (eff_rank / max possible rank),
    ``jacobian_top_sv``, ``jacobian_min_sv``
    """
    J = problem.jacobian(Y)            # (b, m+e, n)
    # SVD per sample
    S = torch.linalg.svdvals(J)        # (b, min(m+e, n))

    # Effective rank per sample
    S_sum = S.sum(dim=1, keepdim=True).clamp(min=1e-12)
    S_norm = S / S_sum                            # normalised
    log_S = torch.log(S_norm.clamp(min=1e-12))
    entropy = -(S_norm * log_S).sum(dim=1)        # (b,)
    eff_rank = torch.exp(entropy)                 # (b,)

    max_rank = float(S.shape[1])

    return {
        "jacobian_eff_rank":    eff_rank.mean().item(),
        "jacobian_rank_ratio":  (eff_rank.mean() / max_rank).item(),
        "jacobian_top_sv":      S[:, 0].mean().item(),
        "jacobian_min_sv":      S[:, -1].mean().item(),
        "jacobian_condition":   (S[:, 0] / S[:, -1].clamp(min=1e-12)).mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MetricsTracker  – convenient wrapper for use in a training loop
# ═══════════════════════════════════════════════════════════════════════════

class MetricsTracker:
    """
    Thin orchestrator that calls the individual metric functions and merges
    the resulting dicts.

    Usage
    -----
    ::

        tracker = MetricsTracker(problem, log_hessian_every=50)

        for step, (X, Y_true) in enumerate(loader):
            Y_pred = model(X)
            loss = problem.get_soft_penalty_loss(Y_pred, X, penalty_weight=10)
            loss.backward()

            metrics = tracker.step(
                model=model,
                problem=problem,
                X=X,
                Y=Y_pred,
                loss=loss,
                step=step,
                loss_fn=lambda: problem.get_soft_penalty_loss(
                    model(X), X, penalty_weight=10,
                ).mean(),
                penalty_weight=10.0,
            )

            wandb.log(metrics, step=step)
            optimizer.step()
            optimizer.zero_grad()

    Parameters
    ----------
    problem            : QCQPProblem instance
    log_hessian_every  : compute Hessian eigenvalues every N steps (0 = never)
    hessian_k          : number of top eigenvalues to track
    log_layer_snr      : include per-layer gradient SNR  (verbose)
    violation_eps      : threshold for counting a constraint as violated
    """

    def __init__(
        self,
        problem,
        log_hessian_every: int = 50,
        hessian_k: int = 3,
        log_layer_snr: bool = False,
        violation_eps: float = 1e-6,
    ):
        self.problem = problem
        self.log_hessian_every = log_hessian_every
        self.hessian_k = hessian_k
        self.log_layer_snr = log_layer_snr
        self.violation_eps = violation_eps

    # ------------------------------------------------------------------

    def step(
        self,
        model: nn.Module,
        problem,
        X: torch.Tensor,
        Y: torch.Tensor,
        loss: torch.Tensor,
        step: int,
        loss_fn=None,
        penalty_weight: float = 1.0,
    ) -> dict:
        """
        Compute all requested metrics for the current training step.

        Call this **after** ``loss.backward()`` so that parameter gradients
        are populated.

        Parameters
        ----------
        model          : the model (gradients must be populated)
        problem        : QCQPProblem
        X              : (b, e)  batch inputs
        Y              : (b, n)  model predictions  (**must still be in the
                         computation graph** for grad_cosine_sim; call
                         ``loss.backward(retain_graph=True)`` if needed)
        loss           : scalar loss tensor (detached or not)
        step           : global training step index
        loss_fn        : callable() → scalar loss for Hessian computation.
                         Only required when ``log_hessian_every > 0``.
        penalty_weight : λ for soft-penalty gradient decomposition

        Returns
        -------
        dict  –  flat dictionary of all metrics
        """
        metrics: dict = {}

        # 1. Loss
        metrics.update(compute_loss(loss))

        # 2 & 3. Constraint violations
        metrics.update(
            compute_constraint_violations(problem, X, Y.detach(), eps=self.violation_eps)
        )

        # 4. Gradient cosine similarity
        if Y.requires_grad or (Y.grad_fn is not None):
            try:
                metrics.update(
                    compute_grad_cosine_similarity(
                        model, problem, X, Y,
                        penalty_weight=penalty_weight,
                    )
                )
            except RuntimeError:
                # Graph may have been freed; skip silently
                pass

        # 5. Gradient SNR
        snr = compute_grad_snr(model)
        if self.log_layer_snr:
            metrics.update(snr)
        else:
            metrics["grad_snr_global"] = snr.get("grad_snr_global", 0.0)

        # 6. Hessian top eigenvalues (periodic)
        if self.log_hessian_every > 0 and loss_fn is not None and step % self.log_hessian_every == 0:
            try:
                metrics.update(
                    compute_hessian_top_eigenvalues(
                        loss_fn, model, k=self.hessian_k,
                    )
                )
            except RuntimeError:
                pass

        # 7. Effective rank of the constraint Jacobian
        metrics.update(compute_jacobian_effective_rank(problem, Y.detach()))

        return metrics
