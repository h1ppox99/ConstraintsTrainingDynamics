#!/usr/bin/env python
"""
Comparative training experiment for the three constraint-handling techniques.

Trains SoftPenaltyNet and TheseusLayerNet on the same QCQP
dataset, logging all training dynamics metrics for comparison.

Usage
-----
    uv run train.py                                  # defaults
    uv run train.py training.epochs=200 training.lr=1e-3   # override
    uv run train.py models=[soft]                    # subset
    uv run train.py wandb.enabled=false              # disable W&B

Config files live in conf/. Results are saved to results/<timestamp>/.
"""

import json
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float64)

from dataset.qcqp_problem import QCQPProblem
from models import SoftPenaltyNet, TheseusLayerNet
from training_dynamics.metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loader(
    X: torch.Tensor,
    opt_vals: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
):
    """Wrap X and opt_vals in a DataLoader (labels come from the problem, not the dataset)."""
    ds = TensorDataset(X, opt_vals)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_model(name: str, problem: QCQPProblem, cfg: dict) -> nn.Module:
    """Instantiate a model by name."""
    backbone_type = cfg.get("backbone_type", "mlp")
    # Backbone kwargs – includes both MLP and Transformer keys; the factory
    # filters to the relevant ones.
    backbone_kwargs = dict(
        hidden_dim=cfg["hidden_dim"],
        n_hidden=cfg["n_hidden"],
        dropout=cfg["dropout"],
        use_batchnorm=cfg["use_batchnorm"],
        d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 3),
        dim_feedforward=cfg.get("dim_feedforward", 128),
        n_tokens=cfg.get("n_tokens", 8),
    )
    if name == "soft":
        return SoftPenaltyNet(
            input_dim=problem.neq,
            output_dim=problem.ydim,
            backbone_type=backbone_type,
            **backbone_kwargs,
        )
    elif name == "theseus":
        return TheseusLayerNet(
            problem,
            backbone_type=backbone_type,
            newton_maxiter=cfg.get("theseus_maxiter", 50),
            **backbone_kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def compute_loss(
    model_name: str,
    problem: QCQPProblem,
    Y: torch.Tensor,
    X: torch.Tensor,
    penalty_weight: float,
    opt_vals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the appropriate scalar loss for a given model type."""
    if model_name == "soft":
        return problem.get_soft_penalty_loss(Y, X, penalty_weight, opt_vals).mean()
    else:
        # Theseus: feasibility built-in → optimality gap
        return problem.get_objective_loss(Y, X, opt_vals).mean()


# ---------------------------------------------------------------------------
# Single-model training loop
# ---------------------------------------------------------------------------

def train_one_model(
    model_name: str,
    model: nn.Module,
    problem: QCQPProblem,
    cfg: dict,
    results_dir: Path,
    use_wandb: bool = False,
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

    train_loader = make_loader(problem.trainX, problem.trainOptvals, batch_size)
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

        for (X_batch, optval_batch) in train_loader:
            optimizer.zero_grad()

            Y_pred = model(X_batch)
            loss = compute_loss(model_name, problem, Y_pred, X_batch, penalty_weight, optval_batch)

            loss.backward(retain_graph=True)

            # --- Metrics (once per epoch, on last batch) ---
            # We collect detailed metrics on the last batch of each epoch
            is_last_batch = (n_batches == len(train_loader) - 1)

            if is_last_batch:
                def loss_fn():
                    return compute_loss(
                        model_name, problem, model(X_batch), X_batch, penalty_weight, optval_batch,
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
            val_obj = problem.get_objective_loss(Y_val, valid_X, problem.validOptvals).mean().item()
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

        # --- Log to wandb ---
        if use_wandb:
            wandb.log(
                {k: v for k, v in record.items()},
                step=global_step,
            )

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

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point – config is fully managed by Hydra (see conf/)."""

    # --- Output directory (set by hydra.run.dir in conf/config.yaml) ---
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")

    # --- Seed ---
    torch.manual_seed(cfg.seed)

    # --- Load dataset ---
    print(f"\nLoading dataset: {cfg.dataset.path}")
    problem = QCQPProblem.load(cfg.dataset.path)
    print(f"  {problem}")
    print(f"  Train: {len(problem.trainX)},  Valid: {len(problem.validX)},  "
          f"Test: {len(problem.testX)}")

    # Build flat cfg for helper functions and backward-compat JSON serialisation
    train_cfg: dict = {
        "epochs":            cfg.training.epochs,
        "batch_size":        cfg.training.batch_size,
        "lr":                cfg.training.lr,
        "penalty_weight":    cfg.training.penalty_weight,
        "hidden_dim":        cfg.model.hidden_dim,
        "n_hidden":          cfg.model.n_hidden,
        "dropout":           cfg.model.dropout,
        "use_batchnorm":     cfg.model.use_batchnorm,
        "log_hessian_every": cfg.metrics.log_hessian_every,
        "hessian_k":         cfg.metrics.hessian_k,
        "theseus_maxiter":   cfg.model.theseus_maxiter,
        "seed":              cfg.seed,
    }

    # Save resolved config as JSON (for plot_comparison.py and post-hoc inspection)
    with open(out_dir / "config.json", "w") as f:
        json.dump(train_cfg, f, indent=2)

    # --- Train each model ---
    all_histories: dict[str, list[dict]] = {}

    for model_name in cfg.models:
        torch.manual_seed(cfg.seed)  # same init for fair comparison

        model = build_model(model_name, problem, train_cfg)
        t0 = time.time()

        # Initialise a per-model W&B run if enabled
        if cfg.wandb.enabled:
            if wandb.run is not None:
                wandb.finish()

            # Build run name: method-agnostic base + method-specific suffix
            if cfg.wandb.run_name:
                run_name = cfg.wandb.run_name
            else:
                lr_str = f"{cfg.training.lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
                backbone_type = cfg.model.backbone_type
                if backbone_type == "transformer":
                    arch_str = f"dm{cfg.model.d_model}x{cfg.model.n_layers}"
                else:
                    arch_str = f"h{cfg.model.hidden_dim}x{cfg.model.n_hidden}"
                run_name = (
                    f"lr{lr_str}"
                    f"_ep{cfg.training.epochs}"
                    f"_bs{cfg.training.batch_size}"
                    f"_{arch_str}"
                )
                if model_name == "soft":
                    run_name += f"_pw{cfg.training.penalty_weight}"
                elif model_name == "theseus":
                    run_name += f"_mi{cfg.model.theseus_maxiter}"

            wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                # Folder hierarchy: method / backbone  (e.g. "soft/mlp")
                group=f"{model_name}/{cfg.model.backbone_type}",
                tags=[
                    model_name,
                    cfg.model.backbone_type,
                    f"ep{cfg.training.epochs}",
                    f"lr{cfg.training.lr}",
                    f"bs{cfg.training.batch_size}",
                ],
                config={**train_cfg, "model_name": model_name},
                dir=str(out_dir),
            )
            print(f"W&B run ({model_name}): {wandb.run.url}")

        try:
            history = train_one_model(
                model_name, model, problem, train_cfg, out_dir,
                use_wandb=cfg.wandb.enabled,
            )
            elapsed = time.time() - t0
            print(f"\n  [{model_name}] Done in {elapsed:.1f}s")
            all_histories[model_name] = history
        except Exception as e:
            print(f"\n  [{model_name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            if cfg.wandb.enabled:
                wandb.finish(exit_code=1)
            continue

        if cfg.wandb.enabled:
            wandb.finish()

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
        torch.manual_seed(cfg.seed)
        model = build_model(model_name, problem, train_cfg)
        ckpt = out_dir / f"{model_name}_checkpoint.pt"
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()

        with torch.no_grad():
            Y_test = model(test_X)
            obj = problem.get_objective_loss(Y_test, test_X, problem.testOptvals).mean().item()
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
        plot_all(all_histories, out_dir, train_cfg)
        print(f"Plots saved to {out_dir}")
    except Exception as e:
        print(f"Plotting failed: {e}  (run plot_comparison.py separately)")

    print(f"\nExperiment complete. Results in: {out_dir}")


if __name__ == "__main__":
    main()
