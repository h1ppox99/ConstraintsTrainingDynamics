#!/usr/bin/env python
"""
Analyze LR sweep results for soft-penalty training (MLP h128×3).

Scans Hydra multirun directories (and standalone results/ folders) for
completed soft-penalty runs with hidden_dim=128 and n_hidden=3, then
produces:
  1. A summary table ranking learning rates per (batch_size, penalty_weight)
  2. Convergence curves grouped by (batch_size, penalty_weight)
  3. A heatmap of the best final val_objective across the LR × BS × PW grid

Usage
-----
    # Auto-detect all multirun directories:
    uv run scripts/analyze_lr_sweep.py

    # Scan a specific multirun sweep:
    uv run scripts/analyze_lr_sweep.py --multirun-dir multirun/2026-02-28/09-04-16

    # Scan standalone results/ folders:
    uv run scripts/analyze_lr_sweep.py --results-dir results

    # Output directory for figures and CSV:
    uv run scripts/analyze_lr_sweep.py --out analysis_lr_sweep
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_runs(search_dirs: list[Path]) -> list[dict]:
    """
    Walk through directories looking for (config.json, histories.json,
    test_results.json) triplets.  Returns a list of dicts, one per valid
    soft-penalty run with hidden_dim=128, n_hidden=3.
    """
    runs = []
    seen_paths = set()

    for root in search_dirs:
        for config_path in root.rglob("config.json"):
            run_dir = config_path.parent
            if str(run_dir) in seen_paths:
                continue
            seen_paths.add(str(run_dir))

            hist_path = run_dir / "histories.json"
            test_path = run_dir / "test_results.json"

            if not hist_path.exists():
                continue

            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                with open(hist_path) as f:
                    histories = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            # Filter: must have a soft model history
            if "soft" not in histories or len(histories["soft"]) == 0:
                continue

            # Filter: MLP h128×3
            if cfg.get("hidden_dim") != 128 or cfg.get("n_hidden") != 3:
                continue

            # Extract key hyperparams
            lr = cfg["lr"]
            bs = cfg["batch_size"]
            pw = cfg["penalty_weight"]

            soft_hist = histories["soft"]

            # Final-epoch metrics
            final = soft_hist[-1]
            val_obj = final["val_objective"]
            val_ineq = final["val_ineq_viol"]
            val_eq = final["val_eq_viol"]
            train_loss = final["train_loss_avg"]

            # Best validation objective across training
            best_val_obj = min(e["val_objective"] for e in soft_hist)
            best_epoch = min(
                soft_hist,
                key=lambda e: e["val_objective"],
            )["epoch"]

            # Test metrics (if available)
            test_obj = None
            test_ineq = None
            test_eq = None
            if test_path.exists():
                try:
                    with open(test_path) as f:
                        test_data = json.load(f)
                    if "soft" in test_data:
                        test_obj = test_data["soft"]["objective_mean"]
                        test_ineq = test_data["soft"]["ineq_viol_mean"]
                        test_eq = test_data["soft"]["eq_viol_mean"]
                except (json.JSONDecodeError, KeyError):
                    pass

            runs.append({
                "dir": str(run_dir),
                "lr": lr,
                "batch_size": bs,
                "penalty_weight": pw,
                "epochs": cfg["epochs"],
                # Final epoch
                "final_val_obj": val_obj,
                "final_val_ineq": val_ineq,
                "final_val_eq": val_eq,
                "final_train_loss": train_loss,
                # Best epoch
                "best_val_obj": best_val_obj,
                "best_epoch": best_epoch,
                # Test
                "test_obj": test_obj,
                "test_ineq": test_ineq,
                "test_eq": test_eq,
                # Full history for curves
                "history": soft_hist,
            })

    return runs


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def group_by(runs: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    """Group runs by a tuple of config keys."""
    groups = defaultdict(list)
    for r in runs:
        key = tuple(r[k] for k in keys)
        groups[key].append(r)
    return dict(groups)


def rank_lr_for_group(group_runs: list[dict]) -> list[dict]:
    """Rank runs within a (bs, pw) group by best_val_obj (lower is better)."""
    return sorted(group_runs, key=lambda r: r["best_val_obj"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(runs: list[dict]) -> str:
    """Print a formatted summary table and return as string."""
    groups = group_by(runs, ["batch_size", "penalty_weight"])
    lines = []

    header = (
        f"{'BS':>4s}  {'PW':>6s}  {'LR':>10s}  "
        f"{'BestValObj':>11s}  {'BestEp':>6s}  "
        f"{'FinalValObj':>11s}  {'FinalIneq':>10s}  {'FinalEq':>10s}  "
        f"{'TestObj':>10s}  {'Rank':>4s}"
    )
    sep = "-" * len(header)

    lines.append("")
    lines.append("=" * len(header))
    lines.append("  LR Sweep Results — Soft Penalty, MLP h128×3")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)

    # Best configs per group
    best_configs = {}

    for (bs, pw) in sorted(groups.keys()):
        ranked = rank_lr_for_group(groups[(bs, pw)])
        best_configs[(bs, pw)] = ranked[0]

        for rank, r in enumerate(ranked, 1):
            marker = " <<<" if rank == 1 else ""
            test_str = f"{r['test_obj']:10.4f}" if r["test_obj"] is not None else "       N/A"
            line = (
                f"{bs:4d}  {pw:6.1f}  {r['lr']:10.1e}  "
                f"{r['best_val_obj']:11.4f}  {r['best_epoch']:6d}  "
                f"{r['final_val_obj']:11.4f}  {r['final_val_ineq']:10.6f}  {r['final_val_eq']:10.6f}  "
                f"{test_str}  {rank:4d}{marker}"
            )
            lines.append(line)
        lines.append(sep)

    # Best per-group summary
    lines.append("")
    lines.append("Best LR per (batch_size, penalty_weight):")
    lines.append(f"{'BS':>4s}  {'PW':>6s}  {'BestLR':>10s}  {'BestValObj':>11s}  {'BestEp':>6s}")
    lines.append("-" * 50)
    for (bs, pw) in sorted(best_configs.keys()):
        r = best_configs[(bs, pw)]
        lines.append(
            f"{bs:4d}  {pw:6.1f}  {r['lr']:10.1e}  {r['best_val_obj']:11.4f}  {r['best_epoch']:6d}"
        )

    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_convergence_curves(runs: list[dict], out_dir: Path):
    """
    For each (batch_size, penalty_weight), plot val_objective vs epoch
    for all learning rates — one subplot per (bs, pw).
    """
    groups = group_by(runs, ["batch_size", "penalty_weight"])
    all_bs = sorted({r["batch_size"] for r in runs})
    all_pw = sorted({r["penalty_weight"] for r in runs})

    n_rows = len(all_bs)
    n_cols = len(all_pw)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        squeeze=False,
        sharex=True,
    )
    fig.suptitle("Soft Penalty — Val Objective vs Epoch (MLP h128×3)", fontsize=14, y=1.01)

    # Color map for learning rates
    all_lrs = sorted({r["lr"] for r in runs})
    colors = cm.viridis(np.linspace(0, 1, len(all_lrs)))
    lr_color = {lr: colors[i] for i, lr in enumerate(all_lrs)}

    for i, bs in enumerate(all_bs):
        for j, pw in enumerate(all_pw):
            ax = axes[i][j]
            key = (bs, pw)
            if key not in groups:
                ax.set_visible(False)
                continue

            group_runs = sorted(groups[key], key=lambda r: r["lr"])
            for r in group_runs:
                epochs = [e["epoch"] for e in r["history"]]
                vals = [e["val_objective"] for e in r["history"]]
                ax.plot(
                    epochs, vals,
                    label=f"lr={r['lr']:.0e}",
                    color=lr_color[r["lr"]],
                    linewidth=1.2,
                    alpha=0.85,
                )

            ax.set_title(f"BS={bs}, λ={pw}", fontsize=11)
            ax.set_ylabel("Val Objective")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "convergence_val_objective.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Same for train loss ---
    fig2, axes2 = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        squeeze=False,
        sharex=True,
    )
    fig2.suptitle("Soft Penalty — Train Loss vs Epoch (MLP h128×3)", fontsize=14, y=1.01)

    for i, bs in enumerate(all_bs):
        for j, pw in enumerate(all_pw):
            ax = axes2[i][j]
            key = (bs, pw)
            if key not in groups:
                ax.set_visible(False)
                continue

            group_runs = sorted(groups[key], key=lambda r: r["lr"])
            for r in group_runs:
                epochs = [e["epoch"] for e in r["history"]]
                losses = [e["train_loss_avg"] for e in r["history"]]
                ax.plot(
                    epochs, losses,
                    label=f"lr={r['lr']:.0e}",
                    color=lr_color[r["lr"]],
                    linewidth=1.2,
                    alpha=0.85,
                )

            ax.set_title(f"BS={bs}, λ={pw}", fontsize=11)
            ax.set_ylabel("Train Loss")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = out_dir / "convergence_train_loss.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # --- Constraint violations ---
    fig3, axes3 = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        squeeze=False,
        sharex=True,
    )
    fig3.suptitle("Soft Penalty — Val Constraint Violations vs Epoch (MLP h128×3)", fontsize=14, y=1.01)

    for i, bs in enumerate(all_bs):
        for j, pw in enumerate(all_pw):
            ax = axes3[i][j]
            key = (bs, pw)
            if key not in groups:
                ax.set_visible(False)
                continue

            group_runs = sorted(groups[key], key=lambda r: r["lr"])
            for r in group_runs:
                epochs = [e["epoch"] for e in r["history"]]
                total_viol = [
                    e["val_ineq_viol"] + e["val_eq_viol"]
                    for e in r["history"]
                ]
                ax.plot(
                    epochs, total_viol,
                    label=f"lr={r['lr']:.0e}",
                    color=lr_color[r["lr"]],
                    linewidth=1.2,
                    alpha=0.85,
                )

            ax.set_title(f"BS={bs}, λ={pw}", fontsize=11)
            ax.set_ylabel("Total Violation (ineq + eq)")
            ax.set_xlabel("Epoch")
            ax.set_yscale("log")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = out_dir / "convergence_violations.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")


def plot_best_lr_heatmap(runs: list[dict], out_dir: Path):
    """
    Heatmap: rows = batch_size, columns = penalty_weight.
    Cell value/color = best val_objective achieved.
    Cell annotation = best LR.
    """
    groups = group_by(runs, ["batch_size", "penalty_weight"])
    all_bs = sorted({r["batch_size"] for r in runs})
    all_pw = sorted({r["penalty_weight"] for r in runs})

    obj_grid = np.full((len(all_bs), len(all_pw)), np.nan)
    lr_grid = np.empty((len(all_bs), len(all_pw)), dtype=object)

    for i, bs in enumerate(all_bs):
        for j, pw in enumerate(all_pw):
            key = (bs, pw)
            if key in groups:
                best = rank_lr_for_group(groups[key])[0]
                obj_grid[i, j] = best["best_val_obj"]
                lr_grid[i, j] = f"{best['lr']:.0e}"

    fig, ax = plt.subplots(figsize=(3 + 2 * len(all_pw), 2 + 1.5 * len(all_bs)))
    im = ax.imshow(obj_grid, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Best Val Objective")

    ax.set_xticks(range(len(all_pw)))
    ax.set_xticklabels([f"λ={pw}" for pw in all_pw])
    ax.set_yticks(range(len(all_bs)))
    ax.set_yticklabels([f"BS={bs}" for bs in all_bs])
    ax.set_title("Best Val Objective (annotated with best LR)\nSoft Penalty, MLP h128×3")

    for i in range(len(all_bs)):
        for j in range(len(all_pw)):
            if lr_grid[i, j] is not None:
                ax.text(j, i, f"{lr_grid[i, j]}\n{obj_grid[i, j]:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if obj_grid[i, j] < np.nanmedian(obj_grid) else "black")

    plt.tight_layout()
    path = out_dir / "best_lr_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_lr_sensitivity(runs: list[dict], out_dir: Path):
    """
    For each penalty_weight, plot best_val_obj vs LR with one line per batch_size.
    Shows how sensitive training is to the learning rate.
    """
    all_bs = sorted({r["batch_size"] for r in runs})
    all_pw = sorted({r["penalty_weight"] for r in runs})
    all_lrs = sorted({r["lr"] for r in runs})

    n_cols = len(all_pw)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
    fig.suptitle("LR Sensitivity — Best Val Objective (MLP h128×3)", fontsize=14)

    markers = ["o", "s", "^", "D", "v", "p"]
    colors_bs = cm.tab10(np.linspace(0, 0.3, len(all_bs)))

    for j, pw in enumerate(all_pw):
        ax = axes[0][j]
        for k, bs in enumerate(all_bs):
            lr_vals = []
            obj_vals = []
            for lr in all_lrs:
                matching = [r for r in runs
                            if r["lr"] == lr and r["batch_size"] == bs
                            and r["penalty_weight"] == pw]
                if matching:
                    best = min(matching, key=lambda r: r["best_val_obj"])
                    lr_vals.append(lr)
                    obj_vals.append(best["best_val_obj"])

            ax.plot(lr_vals, obj_vals,
                    marker=markers[k % len(markers)],
                    label=f"BS={bs}",
                    color=colors_bs[k],
                    linewidth=1.5,
                    markersize=6)

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Best Val Objective")
        ax.set_title(f"λ = {pw}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "lr_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(runs: list[dict], out_dir: Path):
    """Save a CSV with one row per run for further analysis."""
    import csv

    path = out_dir / "lr_sweep_results.csv"
    fields = [
        "batch_size", "penalty_weight", "lr", "epochs",
        "best_val_obj", "best_epoch",
        "final_val_obj", "final_val_ineq", "final_val_eq", "final_train_loss",
        "test_obj", "test_ineq", "test_eq",
        "dir",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(runs, key=lambda r: (r["batch_size"], r["penalty_weight"], r["lr"])):
            writer.writerow(r)

    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze soft-penalty LR sweep results (MLP h128×3)"
    )
    parser.add_argument(
        "--multirun-dir", type=str, default=None,
        help="Path to a specific Hydra multirun directory (e.g. multirun/2026-02-28/09-04-16). "
             "If omitted, scans all multirun/ subdirectories.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to standalone results directory. If omitted, scans results/.",
    )
    parser.add_argument(
        "--out", type=str, default="analysis_lr_sweep",
        help="Output directory for figures and CSV (default: analysis_lr_sweep).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Collect search directories
    search_dirs = []
    if args.multirun_dir:
        search_dirs.append(Path(args.multirun_dir))
    else:
        multirun_root = project_root / "multirun"
        if multirun_root.exists():
            search_dirs.append(multirun_root)

    if args.results_dir:
        search_dirs.append(Path(args.results_dir))
    else:
        results_root = project_root / "results"
        if results_root.exists():
            search_dirs.append(results_root)

    if not search_dirs:
        print("ERROR: No multirun/ or results/ directories found. "
              "Run the sweep first with scripts/sweep_soft_lr_search.sh")
        return

    print(f"Scanning directories: {[str(d) for d in search_dirs]}")
    runs = collect_runs(search_dirs)
    print(f"Found {len(runs)} matching runs (soft, MLP h128×3)")

    if not runs:
        print("\nNo matching runs found. Make sure you have run the LR sweep with:")
        print("  bash scripts/sweep_soft_lr_search.sh")
        print("\nRuns must have hidden_dim=128, n_hidden=3, and a 'soft' history.")
        return

    # Output directory
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    summary = print_summary_table(runs)

    # Save summary to file
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)
    print(f"\n  Saved: {out_dir / 'summary.txt'}")

    # Generate plots
    print("\nGenerating plots...")
    plot_convergence_curves(runs, out_dir)
    plot_best_lr_heatmap(runs, out_dir)
    plot_lr_sensitivity(runs, out_dir)

    # Save CSV
    save_csv(runs, out_dir)

    print(f"\nAnalysis complete. All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
