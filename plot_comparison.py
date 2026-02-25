#!/usr/bin/env python
"""
Comparison plots for the QCQP constraint-handling experiment.

Reads training histories (JSON) and produces multi-panel figures comparing
all three techniques across the 7 metric families.

Can be used standalone:
    uv run python experiments/plot_comparison.py experiments/results/<timestamp>/

Or imported:
    from plot_comparison import plot_all
    plot_all(histories_dict, output_dir, config_dict)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Colour / style config
# ---------------------------------------------------------------------------

MODEL_STYLES = {
    "soft":    {"color": "#e74c3c", "label": "Soft Penalty",    "ls": "-"},
    "theseus": {"color": "#3498db", "label": "Theseus Layer",   "ls": "-"},
}

def _style(name: str):
    return MODEL_STYLES.get(name, {"color": "gray", "label": name, "ls": "--"})


def _extract(history: list[dict], key: str):
    """Extract (epochs, values) for a given metric key, skipping missing."""
    epochs, vals = [], []
    for rec in history:
        if key in rec and rec[key] is not None:
            epochs.append(rec["epoch"])
            vals.append(rec[key])
    return np.array(epochs), np.array(vals)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def plot_loss(ax, histories: dict):
    """Training loss (train_loss_avg) vs epoch."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "train_loss_avg")
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Loss")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_val_objective(ax, histories: dict):
    """Validation objective vs epoch."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "val_objective")
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("Objective")
    ax.set_xlabel("Epoch")
    ax.set_title("Validation Objective")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_constraint_violations(ax_ineq, ax_eq, histories: dict):
    """Mean inequality and equality violations on validation set."""
    for name, hist in histories.items():
        s = _style(name)
        ep, vi = _extract(hist, "val_ineq_viol")
        ax_ineq.plot(ep, vi, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
        ep, ve = _extract(hist, "val_eq_viol")
        ax_eq.plot(ep, ve, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)

    for ax, title in [(ax_ineq, "Ineq. Violation (val)"), (ax_eq, "Eq. Violation (val)")]:
        ax.set_ylabel("Mean Violation")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


def plot_n_violated(ax, histories: dict):
    """Number of violated constraints vs epoch."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "n_total_violated")
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("# Violated")
    ax.set_xlabel("Epoch")
    ax.set_title("Constraints Violated (train batch)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_frac_violated(ax, histories: dict):
    """Fraction of violated constraints vs epoch."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "frac_total_violated")
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Epoch")
    ax.set_title("Fraction Constraints Violated")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_grad_cosine(ax, histories: dict):
    """Gradient cosine similarity (obj ↔ constraint) vs epoch."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "grad_cosine_sim")
        if len(ep) == 0:
            continue
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("Cosine Similarity")
    ax.set_xlabel("Epoch")
    ax.set_title("Grad Cosine (obj ↔ constraint)")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_grad_norms(ax, histories: dict):
    """Gradient norms of objective and violation components."""
    for name, hist in histories.items():
        s = _style(name)
        ep, g_obj = _extract(hist, "grad_obj_norm")
        _, g_viol = _extract(hist, "grad_viol_norm")
        if len(ep) == 0:
            continue
        ax.plot(ep, g_obj, color=s["color"], ls="-", lw=1, alpha=0.7,
                label=f"{s['label']} (obj)")
        ax.plot(ep, g_viol, color=s["color"], ls="--", lw=1, alpha=0.7,
                label=f"{s['label']} (viol)")
    ax.set_ylabel("Gradient Norm")
    ax.set_xlabel("Epoch")
    ax.set_title("Gradient Norms (obj vs viol)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_grad_snr(ax, histories: dict):
    """Global gradient signal-to-noise ratio."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "grad_snr_global")
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("SNR")
    ax.set_xlabel("Epoch")
    ax.set_title("Gradient SNR")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_hessian_eigenvalues(ax, histories: dict):
    """Top Hessian eigenvalues (when available)."""
    any_plotted = False
    for name, hist in histories.items():
        s = _style(name)
        for k_idx in [1, 2, 3]:
            key = f"hessian_eig_{k_idx}"
            ep, v = _extract(hist, key)
            if len(ep) == 0:
                continue
            any_plotted = True
            ls = ["-", "--", ":"][k_idx - 1]
            ax.plot(ep, v, color=s["color"], ls=ls, lw=1.2,
                    label=f"{s['label']} λ{k_idx}")
    if not any_plotted:
        ax.text(0.5, 0.5, "Hessian eigenvalues\nnot computed\n(log_hessian_every=0)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                color="gray")
    ax.set_ylabel("Eigenvalue")
    ax.set_xlabel("Epoch")
    ax.set_title("Hessian Top Eigenvalues")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_jacobian_rank(ax, histories: dict):
    """Effective rank of the constraint Jacobian."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "jacobian_eff_rank")
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("Effective Rank")
    ax.set_xlabel("Epoch")
    ax.set_title("Jacobian Effective Rank")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_jacobian_condition(ax, histories: dict):
    """Condition number (top_sv / min_sv) of constraint Jacobian."""
    for name, hist in histories.items():
        ep, v = _extract(hist, "jacobian_condition")
        s = _style(name)
        ax.plot(ep, v, color=s["color"], ls=s["ls"], label=s["label"], lw=1.5)
    ax.set_ylabel("Condition Number")
    ax.set_xlabel("Epoch")
    ax.set_title("Jacobian Condition Number")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Summary table (text)
# ---------------------------------------------------------------------------

def print_summary_table(test_results: dict):
    """Print a compact comparison table."""
    header = f"{'Model':>10s}  {'Objective':>10s}  {'Ineq Mean':>10s}  {'Eq Mean':>10s}  {'#Ineq Viol':>10s}  {'#Eq Viol':>10s}"
    print(header)
    print("-" * len(header))
    for name, r in test_results.items():
        print(f"{name:>10s}  {r['objective_mean']:10.4f}  "
              f"{r['ineq_viol_mean']:10.6f}  {r['eq_viol_mean']:10.6f}  "
              f"{r['n_ineq_violated']:10d}  {r['n_eq_violated']:10d}")


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_all(
    histories: dict[str, list[dict]],
    output_dir: Path,
    cfg: Optional[dict] = None,
):
    """Generate all comparison plots and save to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Main dashboard (3×4 grid) ----
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle("Training Dynamics Comparison", fontsize=16, y=0.98)

    plot_loss(axes[0, 0], histories)
    plot_val_objective(axes[0, 1], histories)
    plot_constraint_violations(axes[0, 2], axes[0, 3], histories)

    plot_n_violated(axes[1, 0], histories)
    plot_frac_violated(axes[1, 1], histories)
    plot_grad_cosine(axes[1, 2], histories)
    plot_grad_norms(axes[1, 3], histories)

    plot_grad_snr(axes[2, 0], histories)
    plot_hessian_eigenvalues(axes[2, 1], histories)
    plot_jacobian_rank(axes[2, 2], histories)
    plot_jacobian_condition(axes[2, 3], histories)

    # Add config text
    if cfg:
        info = (f"epochs={cfg.get('epochs')}, lr={cfg.get('lr')}, "
                f"batch={cfg.get('batch_size')}, hidden={cfg.get('hidden_dim')}, "
                f"layers={cfg.get('n_hidden')}, λ={cfg.get('penalty_weight')}")
        fig.text(0.5, 0.01, info, ha="center", fontsize=9, color="gray")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    path = output_dir / "dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 2: Focused constraint violation plot ----
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle("Constraint Satisfaction Detail", fontsize=14)

    # Ineq violation over time
    for name, hist in histories.items():
        s = _style(name)
        ep, v = _extract(hist, "ineq_viol_mean")
        ax1.plot(ep, v, color=s["color"], label=s["label"], lw=1.5)
    ax1.set_title("Ineq Violation (train batch)")
    ax1.set_ylabel("Mean Violation")
    ax1.set_xlabel("Epoch")
    ax1.set_yscale("symlog", linthresh=1e-6)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Eq violation over time
    for name, hist in histories.items():
        s = _style(name)
        ep, v = _extract(hist, "eq_viol_mean")
        ax2.plot(ep, v, color=s["color"], label=s["label"], lw=1.5)
    ax2.set_title("Eq Violation (train batch)")
    ax2.set_ylabel("Mean Violation")
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("symlog", linthresh=1e-6)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Ineq viol max
    for name, hist in histories.items():
        s = _style(name)
        ep, v = _extract(hist, "ineq_viol_max")
        ax3.plot(ep, v, color=s["color"], label=s["label"], lw=1.5)
    ax3.set_title("Max Ineq Violation (train batch)")
    ax3.set_ylabel("Max Violation")
    ax3.set_xlabel("Epoch")
    ax3.set_yscale("symlog", linthresh=1e-6)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig2.tight_layout()
    path2 = output_dir / "constraint_detail.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ---- Figure 3: Gradient dynamics ----
    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig3.suptitle("Gradient Dynamics", fontsize=14)

    plot_grad_cosine(ax1, histories)
    plot_grad_snr(ax2, histories)
    plot_grad_norms(ax3, histories)

    fig3.tight_layout()
    path3 = output_dir / "gradient_dynamics.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_comparison.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    # Load histories
    hist_path = results_dir / "histories.json"
    if not hist_path.exists():
        print(f"Error: {hist_path} not found")
        sys.exit(1)

    with open(hist_path) as f:
        histories = json.load(f)

    # Load config
    cfg_path = results_dir / "config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else None

    plot_all(histories, results_dir, cfg)

    # Print test results if available
    test_path = results_dir / "test_results.json"
    if test_path.exists():
        with open(test_path) as f:
            test_results = json.load(f)
        print("\nTest Results:")
        print_summary_table(test_results)


if __name__ == "__main__":
    main()
