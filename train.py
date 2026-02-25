#!/usr/bin/env python
"""
Comparative training experiment for the three constraint-handling techniques.

Trains SoftPenaltyNet, CvxpyLayerNet, and TheseusLayerNet on the same QCQP
dataset, logging all training dynamics metrics for comparison.

Usage
-----
    uv run python experiments/train.py                           # defaults
    uv run python experiments/train.py --epochs 200 --lr 1e-3   # override
    uv run python experiments/train.py --models soft cvxpy       # subset

Results are saved to  experiments/results/<timestamp>/  as JSON + plots.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dataset.qcqp_problem import QCQPProblem
from models.soft_penalty import SoftPenaltyNet
from models.cvxpy_layer import CvxpyLayerNet
from models.theseus_layer import TheseusLayerNet
from training_dynamics.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(X: torch.Tensor, batch_size: int, shuffle: bool = True):
    """Wrap X in a DataLoader (labels come from the problem, not the dataset)."""
    ds = TensorDataset(X)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_model(name: str, problem: QCQPProblem, cfg: dict) -> nn.Module:
    """Instantiate a model by name."""
    common = dict(
        hidden_dim=cfg["hidden_dim"],
        n_hidden=cfg["n_hidden"],
        dropout=cfg["dropout"],
        use_batchnorm=cfg["use_batchnorm"],
    )
    if name == "soft":
        return SoftPenaltyNet(
            input_dim=problem.neq,
            output_dim=problem.ydim,
            **common,
        )
    elif name == "cvxpy":
        return CvxpyLayerNet(problem, **common)
    elif name == "theseus":
        return TheseusLayerNet(
            problem, **common,
            newton_maxiter=cfg.get("theseus_maxiter", 50),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def compute_loss(
    model_name: str,
    problem: QCQPProblem,
    Y: torch.Tensor,
    X: torch.Tensor,
    penalty_weight: float,
) -> torch.Tensor:
    """Compute the appropriate scalar loss for a given model type."""
    if model_name == "soft":
        return problem.get_soft_penalty_loss(Y, X, penalty_weight).mean()
    else:
        # CVXPy / Theseus: feasibility built-in → pure objective loss
        return problem.get_objective_loss(Y, X).mean()


# ---------------------------------------------------------------------------
# Single-model training loop
# ---------------------------------------------------------------------------

def train_one_model(
    model_name: str,
    model: nn.Module,
    problem: QCQPProblem,
    cfg: dict,
    results_dir: Path,
) -> list[dict]:
    """
    Train *model* and return a list of per-epoch metric dicts.
    """
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    penalty_weight = cfg["penalty_weight"]
    log_hessian_every = cfg["log_hessian_every"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tracker = MetricsTracker(
        problem,
        log_hessian_every=log_hessian_every,
        hessian_k=cfg.get("hessian_k", 3),
        log_layer_snr=False,
    )

    train_loader = make_loader(problem.trainX, batch_size)
    valid_X = problem.validX

    history: list[dict] = []
    global_step = 0

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Params:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs:   {epochs}   LR: {lr}   Batch: {batch_size}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (X_batch,) in train_loader:
            optimizer.zero_grad()

            Y_pred = model(X_batch)
            loss = compute_loss(model_name, problem, Y_pred, X_batch, penalty_weight)

            loss.backward(retain_graph=True)

            # --- Metrics (once per epoch, on last batch) ---
            # We collect detailed metrics on the last batch of each epoch
            is_last_batch = (n_batches == len(train_loader) - 1)

            if is_last_batch:
                def loss_fn():
                    return compute_loss(
                        model_name, problem, model(X_batch), X_batch, penalty_weight,
                    )

                train_metrics = tracker.step(
                    model=model,
                    problem=problem,
                    X=X_batch,
                    Y=Y_pred,
                    loss=loss,
                    step=global_step,
                    loss_fn=loss_fn,
                    penalty_weight=penalty_weight,
                )

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        scheduler.step()

        # --- Validation metrics ---
        model.eval()
        with torch.no_grad():
            Y_val = model(valid_X)
            val_obj = problem.get_objective_loss(Y_val, valid_X).mean().item()
            val_ineq = problem.get_ineq_res(valid_X, Y_val).mean().item()
            val_eq = problem.get_eq_res(valid_X, Y_val).mean().item()

        # Build epoch record
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_avg": epoch_loss / n_batches,
            "val_objective": val_obj,
            "val_ineq_viol": val_ineq,
            "val_eq_viol": val_eq,
            "lr": scheduler.get_last_lr()[0],
        }
        record.update(train_metrics)
        history.append(record)

        # --- Print progress ---
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(
                f"  [{model_name}] Epoch {epoch:4d}/{epochs}  "
                f"loss={record['train_loss_avg']:.4f}  "
                f"val_obj={val_obj:.4f}  "
                f"ineq={val_ineq:.4f}  eq={val_eq:.4f}  "
                f"cosine={record.get('grad_cosine_sim', float('nan')):.3f}  "
                f"snr={record.get('grad_snr_global', 0):.4f}"
            )

    # Save model checkpoint
    ckpt_path = results_dir / f"{model_name}_checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QCQP comparison experiment")

    # Dataset
    parser.add_argument("--dataset", type=str,
                        default="experiments/dataset/data/qcqp_var50_ineq20_eq20_N2000_seed2025.pkl",
                        help="Path to saved QCQPProblem pickle")

    # Models to train
    parser.add_argument("--models", nargs="+", default=["soft", "cvxpy", "theseus"],
                        choices=["soft", "cvxpy", "theseus"],
                        help="Which models to train")

    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--penalty_weight", type=float, default=10.0,
                        help="λ for soft-penalty loss")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--n_hidden", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_batchnorm", action="store_true", default=True)

    # Metrics
    parser.add_argument("--log_hessian_every", type=int, default=0,
                        help="Compute Hessian eigenvalues every N global steps (0=off)")
    parser.add_argument("--hessian_k", type=int, default=3)

    # Theseus
    parser.add_argument("--theseus_maxiter", type=int, default=30)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # --- Seed ---
    torch.manual_seed(args.seed)

    # --- Output directory ---
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = SCRIPT_DIR / "results" / ts
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.dataset}")
    problem = QCQPProblem.load(args.dataset)
    print(f"  {problem}")
    print(f"  Train: {len(problem.trainX)},  Valid: {len(problem.validX)},  "
          f"Test: {len(problem.testX)}")

    # --- Config dict ---
    cfg = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "penalty_weight": args.penalty_weight,
        "hidden_dim": args.hidden_dim,
        "n_hidden": args.n_hidden,
        "dropout": args.dropout,
        "use_batchnorm": args.use_batchnorm,
        "log_hessian_every": args.log_hessian_every,
        "hessian_k": args.hessian_k,
        "theseus_maxiter": args.theseus_maxiter,
        "seed": args.seed,
    }

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # --- Train each model ---
    all_histories: dict[str, list[dict]] = {}

    for model_name in args.models:
        torch.manual_seed(args.seed)  # same init for fair comparison

        model = build_model(model_name, problem, cfg)
        t0 = time.time()

        try:
            history = train_one_model(model_name, model, problem, cfg, out_dir)
            elapsed = time.time() - t0
            print(f"\n  [{model_name}] Done in {elapsed:.1f}s")
            all_histories[model_name] = history
        except Exception as e:
            print(f"\n  [{model_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    # --- Save all histories ---
    hist_path = out_dir / "histories.json"
    with open(hist_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"\nHistories saved to {hist_path}")

    # --- Final test-set evaluation ---
    print(f"\n{'='*60}")
    print("  Final test-set evaluation")
    print(f"{'='*60}")

    test_X = problem.testX
    test_results = {}

    for model_name in all_histories:
        torch.manual_seed(args.seed)
        model = build_model(model_name, problem, cfg)
        ckpt = out_dir / f"{model_name}_checkpoint.pt"
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()

        with torch.no_grad():
            Y_test = model(test_X)
            obj = problem.get_objective_loss(Y_test, test_X).mean().item()
            ineq = problem.get_ineq_res(test_X, Y_test)
            eq = problem.get_eq_res(test_X, Y_test)

        test_results[model_name] = {
            "objective_mean": obj,
            "ineq_viol_mean": ineq.mean().item(),
            "ineq_viol_max": ineq.max().item(),
            "eq_viol_mean": eq.mean().item(),
            "eq_viol_max": eq.max().item(),
            "n_ineq_violated": int((ineq > 1e-6).sum().item()),
            "n_eq_violated": int((eq > 1e-6).sum().item()),
        }

        print(f"  [{model_name:8s}]  obj={obj:10.4f}  "
              f"ineq_mean={ineq.mean().item():.6f}  "
              f"eq_mean={eq.mean().item():.6f}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # --- Generate plots ---
    print(f"\nGenerating comparison plots...")
    try:
        from plot_comparison import plot_all
        plot_all(all_histories, out_dir, cfg)
        print(f"Plots saved to {out_dir}")
    except Exception as e:
        print(f"Plotting failed: {e}  (run plot_comparison.py separately)")

    print(f"\nExperiment complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
