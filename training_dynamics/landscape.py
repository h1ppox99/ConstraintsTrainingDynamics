"""
Loss-landscape visualization with **filter-wise normalization**.

Generates 2-D contour plots and 3-D surface plots of the loss around a
model checkpoint.  The implementation follows the approach of

    Li et al., "Visualizing the Loss Landscape of Neural Nets", NeurIPS 2018

and reuses the normalisation logic from the ``loss-landscape`` submodule
(Tom Goldstein).

Public API
----------
``generate_landscape``
    All-in-one helper that produces a 2D + 3D .png for one model at the
    current weights.  Call it from the training loop.

``LandscapeConfig``
    Typed container for the knobs exposed via Hydra (grid resolution,
    coordinate range, etc.).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for cluster jobs

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection


# ---------------------------------------------------------------------------
# Config dataclass (filled from Hydra)
# ---------------------------------------------------------------------------

@dataclass
class LandscapeConfig:
    """All tunables for the landscape scan."""

    enabled: bool = False
    every_n_epochs: int = 50          # 0  = only at end
    grid_size: int = 21               # points per axis  → grid_size²  evaluations
    coord_range: float = 1.0          # α, β ∈ [-coord_range, +coord_range]
    norm: str = "filter"              # "filter" | "layer" | "weight"
    ignore: str = "biasbn"            # zero-out 1-D params (bias, BN γ/β)
    log_scale: bool = True            # plot log(1+Z) to compress exponential edge growth
    clip_percentile: float = 95.0     # cap colorscale at this percentile (0 = no cap)


# ---------------------------------------------------------------------------
# Filter-wise normalisation  (matches loss-landscape/net_plotter.py)
# ---------------------------------------------------------------------------

def _normalize_direction(direction: torch.Tensor, weights: torch.Tensor,
                         norm: str = "filter") -> None:
    """
    Rescale *direction* in-place so that its scale matches *weights*.

    For ``norm='filter'`` the first axis is iterated (one "filter"
    per output channel / neuron) and each slice is rescaled to have
    the same Frobenius norm as the corresponding slice in *weights*.
    """
    if norm == "filter":
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == "layer":
        direction.mul_(weights.norm() / (direction.norm() + 1e-10))
    elif norm == "weight":
        direction.mul_(weights)
    else:
        raise ValueError(f"Unknown norm: {norm!r}")


def _create_random_direction(model: nn.Module, norm: str = "filter",
                             ignore: str = "biasbn") -> list[torch.Tensor]:
    """
    Sample a random direction in parameter space and normalise it.

    Returns a list of tensors *aligned with* ``list(model.parameters())``.
    """
    direction: list[torch.Tensor] = []
    for p in model.parameters():
        d = torch.randn_like(p)
        if d.dim() <= 1:
            # Bias / BN affine: zero out so they don't affect the scan
            if ignore == "biasbn":
                d.zero_()
            else:
                d.copy_(p.data)
        else:
            _normalize_direction(d, p.data, norm)
        direction.append(d)
    return direction


# ---------------------------------------------------------------------------
# Model perturbation
# ---------------------------------------------------------------------------

def _set_weights(model: nn.Module,
                 origin: list[torch.Tensor],
                 direction_x: list[torch.Tensor],
                 direction_y: list[torch.Tensor],
                 alpha: float,
                 beta: float) -> None:
    """Set model parameters to  ``origin + α·δ + β·η``  (in-place)."""
    for p, w0, dx, dy in zip(model.parameters(), origin, direction_x, direction_y):
        p.data.copy_(w0 + alpha * dx + beta * dy)


def _restore_weights(model: nn.Module, origin: list[torch.Tensor]) -> None:
    """Restore model parameters to their original values."""
    for p, w0 in zip(model.parameters(), origin):
        p.data.copy_(w0)


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_grid(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    origin: list[torch.Tensor],
    dir_x: list[torch.Tensor],
    dir_y: list[torch.Tensor],
    coords: np.ndarray,
) -> np.ndarray:
    """
    Evaluate *loss_fn* at every (α, β) grid point.

    The model is temporarily perturbed; original weights are restored
    afterwards.  ``model.eval()`` is called to freeze BatchNorm statistics.

    Returns
    -------
    Z : ndarray of shape ``(len(coords), len(coords))``
    """
    model.eval()  # freeze BN running stats, disable dropout
    n = len(coords)
    Z = np.empty((n, n), dtype=np.float64)

    for i, alpha in enumerate(coords):
        for j, beta in enumerate(coords):
            _set_weights(model, origin, dir_x, dir_y, float(alpha), float(beta))
            loss_val = loss_fn().item()
            Z[i, j] = loss_val

    # Always restore
    _restore_weights(model, origin)
    model.train()
    return Z


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_2d_contour(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     title: str, path: Path,
                     vmax: float | None = None,
                     log_scale: bool = False,
                     clip_percentile: float = 95.0) -> None:
    """Save a high-resolution filled contour plot."""
    fig, ax = plt.subplots(figsize=(6, 5))

    Zp = np.log1p(Z) if log_scale else Z.copy()
    loss_label = "log(1 + Loss)" if log_scale else "Loss"

    # Cap colorscale at the requested percentile to avoid edge values dominating
    if vmax is None and clip_percentile > 0:
        vmax = float(np.percentile(Zp, clip_percentile))
    vmin = float(Zp.min())
    if vmax is None or vmax <= vmin:
        vmax = float(Zp.max())
    levels = np.linspace(vmin, vmax, 35)

    cs = ax.contourf(X, Y, Zp, levels=levels, cmap="RdYlBu_r", extend="max")
    ax.contour(X, Y, Zp, levels=levels, linewidths=0.4, colors="k", alpha=0.3)
    fig.colorbar(cs, ax=ax, label=loss_label)

    # Mark centre (original weights)
    ax.plot(0, 0, marker="*", color="black", markersize=12, zorder=5)

    ax.set_xlabel(r"$\alpha$  (direction $\delta$)")
    ax.set_ylabel(r"$\beta$   (direction $\eta$)")
    scale_tag = "  [log scale]" if log_scale else ""
    ax.set_title(title + scale_tag, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_3d_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     title: str, path: Path,
                     vmax: float | None = None,
                     log_scale: bool = False,
                     clip_percentile: float = 95.0) -> None:
    """Save a 3-D surface plot."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    Zp = np.log1p(Z) if log_scale else Z.copy()
    loss_label = "log(1 + Loss)" if log_scale else "Loss"

    if vmax is None and clip_percentile > 0:
        vmax = float(np.percentile(Zp, clip_percentile))
    if vmax is not None:
        Zp = np.clip(Zp, None, vmax)

    surf = ax.plot_surface(X, Y, Zp, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=0.9,
                           rcount=100, ccount=100)
    fig.colorbar(surf, ax=ax, shrink=0.5, label=loss_label)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.set_zlabel(loss_label)
    scale_tag = "  [log scale]" if log_scale else ""
    ax.set_title(title + scale_tag, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_landscape(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    model_name: str,
    epoch: int,
    out_dir: Path,
    lc: LandscapeConfig,
    seed: int | None = None,
) -> dict[str, Path]:
    """
    Generate 2-D contour and 3-D surface loss-landscape plots.

    Parameters
    ----------
    model : nn.Module
        The model at its current checkpoint.
    loss_fn : callable
        Zero-argument callable that returns the scalar loss at the
        **current** model parameters.  Typically wraps a forward pass
        over a fixed mini-batch (or the full training set).
    model_name : str
        e.g. ``"soft"``, ``"theseus"``, ``"cvxpy"``.
    epoch : int
        Current epoch number (used in the file names).
    out_dir : Path
        Experiment results directory.  Images land in
        ``out_dir / landscapes /``.
    lc : LandscapeConfig
        Landscape hyperparameters (grid size, range, norm, …).
    seed : int or None
        If given, seed the RNG to make the directions reproducible.

    Returns
    -------
    dict mapping ``"2d"`` / ``"3d"`` to the saved file paths.
    """
    landscape_dir = out_dir / "landscapes"
    landscape_dir.mkdir(parents=True, exist_ok=True)

    # --- Deterministic directions (same δ, η for 2-D and 3-D) ---
    if seed is not None:
        torch.manual_seed(seed + epoch)

    dir_x = _create_random_direction(model, norm=lc.norm, ignore=lc.ignore)
    dir_y = _create_random_direction(model, norm=lc.norm, ignore=lc.ignore)

    # Snapshot original weights
    origin = [p.data.clone() for p in model.parameters()]

    # --- Build evaluation grid ---
    coords = np.linspace(-lc.coord_range, lc.coord_range, lc.grid_size)
    X, Y = np.meshgrid(coords, coords)

    # --- Evaluate ---
    Z = _evaluate_grid(model, loss_fn, origin, dir_x, dir_y, coords)

    # --- Plot ---
    tag = f"{model_name}_epoch_{epoch}"
    title_base = f"{model_name}  –  epoch {epoch}"

    path_2d = landscape_dir / f"landscape_2d_{tag}.png"
    path_3d = landscape_dir / f"landscape_3d_{tag}.png"

    _plot_2d_contour(X, Y, Z, f"Loss Landscape (2-D)  ·  {title_base}", path_2d,
                     log_scale=lc.log_scale, clip_percentile=lc.clip_percentile)
    _plot_3d_surface(X, Y, Z, f"Loss Landscape (3-D)  ·  {title_base}", path_3d,
                     log_scale=lc.log_scale, clip_percentile=lc.clip_percentile)

    print(f"  [{model_name}] Landscape plots saved → {landscape_dir}")

    return {"2d": path_2d, "3d": path_3d}
