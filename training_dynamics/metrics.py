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

Metrics computed
----------------
1.  loss                  – scalar training loss
2.  constraint_violations – per-type (ineq / eq) violation statistics  (improved)
3.  n_violated            – number of violated constraints
4.  grad_cosine_sim       – cosine similarity + optional PCGrad projection
                            (projects obj gradient to remove conflict with the
                             constraint gradient when similarity < 0)
5.  grad_snr              – global SNR + per-layer SNR + min-layer bottleneck
6.  hessian_top_eig       – top-k eigenvalues + per-eigenvalue residual/
                            convergence flag (stochastic power iteration)
7.  jacobian_eff_rank     – effective rank of the constraint Jacobian
8.  feature_jacobian       – effective rank of the feature Jacobian ∂y/∂x
                            (detects representation collapse independent of
                             constraint satisfaction)

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
    apply_pcgrad: bool = False,
) -> dict:
    """
    Cosine similarity between the gradient of the objective loss and the
    gradient of the constraint-violation penalty, with optional PCGrad
    correction.

    cos(∇_θ L_obj, ∇_θ L_constr)  ∈ [-1, 1]

    A value near **+1** means both losses push parameters in the same
    direction;  **-1** means they are in conflict;  **0** signals
    orthogonality.

    PCGrad (``apply_pcgrad=True``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``cos < 0`` the gradient of the objective is projected onto the
    normal plane of the constraint gradient so that the two tasks no longer
    conflict.  The model's ``.grad`` tensors are overwritten in-place with
    the combined PCGrad vector  ``g_obj_proj + g_viol``.  Call this function
    **after** ``loss.backward()`` and **before** ``optimizer.step()``.

        g_obj_proj = g_obj − (g_obj · g_viol / ‖g_viol‖²) · g_viol

    Parameters
    ----------
    model          : the network whose parameters carry gradients
    problem        : QCQPProblem
    X, Y           : current mini-batch  (Y = model output, must retain grad graph)
    penalty_weight : λ used in the penalty loss (scale-invariant for cosine)
    apply_pcgrad   : if True and cos < 0, overwrite model gradients with the
                     PCGrad-projected combined gradient

    Returns
    -------
    dict with ``grad_cosine_sim``, ``grad_obj_norm``, ``grad_viol_norm``,
    and ``pcgrad_applied`` (bool flag, only present when apply_pcgrad=True)
    """
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    dtype  = params[0].dtype  if params else torch.float64

    # --- Gradient of objective term ---
    obj_loss = problem.get_objective_loss(Y, X).mean()
    obj_grads = torch.autograd.grad(
        obj_loss, params, retain_graph=True, allow_unused=True,
    )
    # Use zeros for parameters with no gradient so flat vectors stay aligned
    g_obj = torch.cat([
        g.flatten() if g is not None else torch.zeros(p.numel(), device=device, dtype=dtype)
        for g, p in zip(obj_grads, params)
    ])

    # --- Gradient of constraint-violation term ---
    resid = problem.get_resid(X, Y)
    viol_loss = (resid ** 2).sum(dim=1).mean()
    viol_grads = torch.autograd.grad(
        viol_loss, params, retain_graph=True, allow_unused=True,
    )
    g_viol = torch.cat([
        g.flatten() if g is not None else torch.zeros(p.numel(), device=device, dtype=dtype)
        for g, p in zip(viol_grads, params)
    ])

    cos = torch.nn.functional.cosine_similarity(
        g_obj.unsqueeze(0), g_viol.unsqueeze(0),
    ).item()

    out = {
        "grad_cosine_sim": cos,
        "grad_obj_norm":   g_obj.norm().item(),
        "grad_viol_norm":  g_viol.norm().item(),
    }

    # --- Optional PCGrad correction ---
    if apply_pcgrad:
        pcgrad_applied = cos < 0
        if pcgrad_applied:
            # Project g_obj onto the normal plane of g_viol
            dot = g_obj.dot(g_viol)
            g_viol_sq = g_viol.dot(g_viol).clamp(min=1e-12)
            g_obj_proj = g_obj - (dot / g_viol_sq) * g_viol
            g_combined = g_obj_proj + g_viol

            # Write back into model .grad tensors (in-place)
            offset = 0
            for p in params:
                n = p.numel()
                if p.grad is not None:
                    p.grad.copy_(g_combined[offset : offset + n].view_as(p))
                offset += n

        out["pcgrad_applied"] = int(pcgrad_applied)

    return out


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
        return {"grad_snr_global": 0.0, "grad_snr_min": 0.0}

    g_all = torch.cat(all_grads)
    global_snr = (g_all.mean().abs() / (g_all.std() + 1e-12)).item()

    # Bottleneck detection: the layer with the lowest SNR corrupts all
    # upstream gradient signals — track it explicitly regardless of whether
    # per-layer logging is enabled.
    if layer_snrs:
        min_layer_name = min(layer_snrs, key=layer_snrs.get)
        grad_snr_min   = layer_snrs[min_layer_name]
    else:
        min_layer_name = ""
        grad_snr_min   = global_snr

    out = {
        "grad_snr_global":    global_snr,
        "grad_snr_min":       grad_snr_min,
        "grad_snr_min_layer": min_layer_name,   # informational; may be filtered by some loggers
    }
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
    dict with
    - ``hessian_eig_1 .. hessian_eig_k``  – estimated eigenvalues
    - ``hessian_eig_max``                  – largest of the k eigenvalues
    - ``hessian_residual_1 .. _k``         – ‖Hv − λv‖ for each eigenpair;
      a high residual indicates the estimate is unreliable (e.g. two
      eigenvalues are nearly degenerate and the iteration oscillated)
    - ``hessian_converged``                – 1 if *all* residuals are below
      ``tol * max(|λ|, 1)``; 0 otherwise.  Do not use ``hessian_eig_max``
      for learning-rate tuning when this flag is 0.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    device = params[0].device
    dtype = params[0].dtype

    eigenvalues: list[float] = []
    residuals:   list[float] = []
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

        # Deflation stability check: compute the true residual ‖Hv − λv‖.
        # A large residual means two eigenvalues are nearly degenerate and the
        # iteration oscillated — the reported λ should not be trusted for
        # learning-rate tuning.  We pay one extra HVP call per eigenvalue.
        with torch.no_grad():
            Hv_raw = _hvp(loss_fn, params, v.detach())
        residual = (Hv_raw - lam * v).norm().item()

        eigenvalues.append(lam)
        residuals.append(residual)
        eigenvectors.append(v.detach())

    converged = all(
        residuals[i] < tol * max(abs(eigenvalues[i]), 1.0) for i in range(k)
    )

    out = {f"hessian_eig_{i+1}": eigenvalues[i] for i in range(k)}
    out.update({f"hessian_residual_{i+1}": residuals[i] for i in range(k)})
    out["hessian_eig_max"]  = max(eigenvalues)
    out["hessian_converged"] = int(converged)
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
#  8.  Effective rank of the feature Jacobian  ∂y/∂x
# ═══════════════════════════════════════════════════════════════════════════

def compute_feature_jacobian_effective_rank(
    model: nn.Module,
    X: torch.Tensor,
) -> dict:
    """
    Effective rank of the feature Jacobian  J_x = ∂y/∂x  ∈ ℝ^{n×e}.

    Complements :func:`compute_jacobian_effective_rank` which measures
    constraint independence.  **If the constraint Jacobian rank is high but
    the feature Jacobian rank is low**, the model is attempting to satisfy
    constraints in a space where it has insufficient expressive power.

    The Jacobian is assembled column-by-column via back-prop  (one
    ``autograd.grad`` call per output dimension *n*).  For large *n* this
    can be expensive; consider calling it periodically rather than every step.

    Parameters
    ----------
    model : nn.Module  (called internally with ``X``) 
    X     : (b, e)  input batch

    Returns
    -------
    dict with
    - ``feature_jacobian_eff_rank``   – batch-averaged effective rank
    - ``feature_jacobian_rank_ratio`` – eff_rank / min(n, e)
    - ``feature_jacobian_top_sv``     – mean largest singular value
    - ``feature_jacobian_min_sv``     – mean smallest singular value
    - ``feature_jacobian_condition``  – mean condition number  σ_max / σ_min
    """
    X_ = X.detach().requires_grad_(True)
    with torch.enable_grad():
        Y_ = model(X_)   # (b, n)
        b, n = Y_.shape

    # Build J ∈ ℝ^{b × n × e} column-by-column
    J_rows: list[torch.Tensor] = []
    for j in range(n):
        g = torch.autograd.grad(
            Y_[:, j].sum(), X_,
            retain_graph=(j < n - 1),
            create_graph=False,
        )[0]           # (b, e)
        J_rows.append(g)

    J = torch.stack(J_rows, dim=1)   # (b, n, e)
    S = torch.linalg.svdvals(J)      # (b, min(n, e))

    S_sum  = S.sum(dim=1, keepdim=True).clamp(min=1e-12)
    S_norm = S / S_sum
    log_S  = torch.log(S_norm.clamp(min=1e-12))
    entropy  = -(S_norm * log_S).sum(dim=1)   # (b,)
    eff_rank = torch.exp(entropy)              # (b,)

    max_rank = float(S.shape[1])

    return {
        "feature_jacobian_eff_rank":   eff_rank.mean().item(),
        "feature_jacobian_rank_ratio": (eff_rank.mean() / max_rank).item(),
        "feature_jacobian_top_sv":     S[:, 0].mean().item(),
        "feature_jacobian_min_sv":     S[:, -1].mean().item(),
        "feature_jacobian_condition":  (S[:, 0] / S[:, -1].clamp(min=1e-12)).mean().item(),
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
    problem              : QCQPProblem instance
    log_hessian_every    : compute Hessian eigenvalues every N steps (0 = never)
    hessian_k            : number of top eigenvalues to track
    log_layer_snr        : include per-layer gradient SNR  (verbose)
    violation_eps        : threshold for counting a constraint as violated
    apply_pcgrad         : when True, apply PCGrad correction to model gradients
                           whenever grad_cosine_sim < 0  (modifies .grad in place;
                           call before optimizer.step())
    log_feature_jacobian : compute the feature Jacobian ∂y/∂x every N steps
                           (0 = never).  Useful for detecting representation
                           collapse independent of constraint satisfaction.
    """

    def __init__(
        self,
        problem,
        log_hessian_every: int = 50,
        hessian_k: int = 3,
        log_layer_snr: bool = False,
        violation_eps: float = 1e-6,
        apply_pcgrad: bool = False,
        log_feature_jacobian: int = 0,
    ):
        self.problem = problem
        self.log_hessian_every = log_hessian_every
        self.hessian_k = hessian_k
        self.log_layer_snr = log_layer_snr
        self.violation_eps = violation_eps
        self.apply_pcgrad = apply_pcgrad
        self.log_feature_jacobian = log_feature_jacobian

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

        # 4. Gradient cosine similarity (+ optional PCGrad correction)
        if Y.requires_grad or (Y.grad_fn is not None):
            try:
                metrics.update(
                    compute_grad_cosine_similarity(
                        model, problem, X, Y,
                        penalty_weight=penalty_weight,
                        apply_pcgrad=self.apply_pcgrad,
                    )
                )
            except RuntimeError:
                # Graph may have been freed; skip silently
                pass

        # 5. Gradient SNR
        #    Always surface the global SNR and the bottleneck-layer min SNR.
        #    Per-layer detail is opt-in via log_layer_snr.
        snr = compute_grad_snr(model)
        metrics["grad_snr_global"]    = snr.get("grad_snr_global", 0.0)
        metrics["grad_snr_min"]       = snr.get("grad_snr_min", 0.0)
        metrics["grad_snr_min_layer"] = snr.get("grad_snr_min_layer", "")
        if self.log_layer_snr:
            metrics.update({
                k: v for k, v in snr.items()
                if k.startswith("grad_snr/")
            })

        # 6. Hessian top eigenvalues + residuals (periodic)
        #    hessian_converged == 0 means the residual is high and λ_max
        #    should not be used for learning-rate tuning.
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

        # 8. Effective rank of the feature Jacobian ∂y/∂x  (periodic)
        #    Cross-reference with constraint Jacobian rank: a high constraint
        #    rank but low feature rank means the model lacks expressive power.
        if self.log_feature_jacobian > 0 and step % self.log_feature_jacobian == 0:
            try:
                metrics.update(
                    compute_feature_jacobian_effective_rank(model, X)
                )
            except RuntimeError:
                pass

        return metrics
