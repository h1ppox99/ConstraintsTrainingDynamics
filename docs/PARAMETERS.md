## Training CLI reference

Configuration is managed by [Hydra](https://hydra.cc/).  

```bash
# Default run
uv run train.py

# Quick override examples
uv run train.py training.epochs=300 training.lr=3e-4
uv run train.py models=[soft]
uv run train.py wandb.enabled=false
```

---

### Top-level

| Parameter | Default | Description |
|-----------|---------|-------------|
| `models` | `[soft, theseus]` | List of models to train. Subset with e.g. `models=[soft]` or `models=[theseus]`. |
| `seed` | `42` | Global random seed for reproducibility. |

---

### `training.*`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | `150` | Number of training epochs. |
| `training.batch_size` | `64` | Mini-batch size. |
| `training.lr` | `1e-3` | Initial learning rate for Adam (cosine-annealed to 0). |
| `training.penalty_weight` | `10.0` | λ — weight of the constraint-violation term in the soft-penalty loss. |

---

### `model.*`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.hidden_dim` | `200` | Width of each hidden layer of the MLP backbone. |
| `model.n_hidden` | `3` | Number of hidden layers. |
| `model.dropout` | `0.1` | Dropout probability (applied after each hidden layer). |
| `model.use_batchnorm` | `true` | Whether to apply batch normalisation in the backbone. |
| `model.theseus_maxiter` | `30` | Maximum Newton iterations for the Theseus projection layer. |

---

### `dataset.*`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset.path` | `dataset/data/qcqp_var50_ineq20_eq20_N2000_seed2025.pkl` | Path to the pre-generated `.pkl` dataset file. |

---

### `metrics.*`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metrics.log_hessian_every` | `0` | Log top-k Hessian eigenvalues every N global steps. Set to `0` to disable (expensive). |
| `metrics.hessian_k` | `3` | Number of leading Hessian eigenvalues to compute when Hessian logging is enabled. |

---

### `wandb.*`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wandb.enabled` | `true` | Enable / disable Weights & Biases logging. |
| `wandb.project` | `ConstraintsTrainingDynamics` | W&B project name. |
| `wandb.run_name` | `null` | W&B run name. When `null`, auto-generated as `ep{epochs}_lr{lr}_h{dim}x{layers}_pw{penalty}`. |

---

### Multirun (hyperparameter sweeps)

Hydra's `--multirun` flag runs all combinations of comma-separated values:

```bash
uv run train.py --multirun \
    training.lr=1e-3,3e-4 \
    training.penalty_weight=1.0,10.0,100.0
```

Results are saved under `multirun/<date>/<time>/<job_id>/`.