# Training Dynamics Metrics

The following metrics are tracked to analyze learning behaviors (from basic to advanced). See also [CONTEXT.md](../CONTEXT.md) for the broader project context.

---

## 1. Performance Metrics

| Metric | Purpose |
| :--- | :--- |
| Loss quantities | Evaluate overall training performance for given configurations |
| Constraint violations (count & magnitude) | Evaluate constraint satisfaction performance |

---

## 2. Gradient Metrics

| Metric | Purpose |
| :--- | :--- |
| **Gradient cosine similarity** | Measures alignment between loss components in the soft penalty setting |
| **Gradient signal-to-noise ratio (SNR)** | Evaluates quality of the gradient signal |
| **Gradient noise scale** | Identifies useful rules for choosing efficient batch sizes (McCandlish et al., 2018) |
| **Jacobian effective rank** | Identifies potential signal collapse in hard constraint settings |

---

## 3. Loss Landscape Metrics

| Metric | Purpose |
| :--- | :--- |
| **Hessian top eigenvalues** | Evaluates loss sharpness; critical for understanding edge-of-stability behavior (Cohen et al., 2022) |
| **2D/3D loss landscape visualizations** | Plots the loss landscape along the most significant directions (Li et al., 2018) |
