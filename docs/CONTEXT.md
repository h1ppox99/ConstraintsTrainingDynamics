# Project Context: Safe & Constrained AI

**Author:** Hippolyte Wallaert, Stanford ICME 

---

## 1. Project Overview & Goals

This project studies the **training dynamics and loss landscapes** of differentiable optimization layers — specifically **cvxpy layers** (Agrawal et al., 2019) and **Theseus layers** (Pineda et al., 2023) — used to enforce constraints in neural networks. The study compares these **hard-constraint** approaches against traditional **soft penalty** techniques to:

- Identify meaningful patterns in the loss landscape induced by differentiable layers.
- Design better training strategies for such layers.
- Enhance the learning abilities of these layers, which already have universal approximation guarantees in simple cases (Min & Azizan, 2025).

The project also establishes that specific repair layers such as **SnareNet** (Chu et al., 2026) can be rewritten as special cases of general Theseus layers (see Appendix section below).

Results are presented through a **modular, web-based visual interface** inspired by LossLens (Xie et al., 2024).

---

## 2. Problem Formulation

Given inputs $x \in \mathcal{X} \subset \mathbb{R}^d$ and outputs $y \in \mathcal{Y} \subset \mathbb{R}^n$, the constrained learning problem is:

$$\min_{\theta} \; \mathbb{E}_{x \sim \mathcal{X}}[\ell(\theta; x)] \quad \text{s.t.} \quad y \in \mathcal{C}_x, \; \forall x \in \mathcal{X}$$

where the constraint set is:

$$\mathcal{C}_x = \{y \in \mathcal{Y} \mid l_x \le g_x(y) \le u_x\}$$

Notations follow Chu et al. (2026).

---

## 3. Constraint Enforcement Techniques

### 3.1 Soft Penalty Loss

The architecture consists of a neural network backbone of any architecture. Constraints are enforced through the loss function:

$$\ell_{\text{soft}}(\theta; x) = \ell(\theta; x) + \mu_u \| \text{ReLU}(g(\hat{y}) - u) \|_2 + \mu_\ell \| \text{ReLU}(\ell - g(\hat{y})) \|_2$$

This strategy offers **no strict guarantee** of constraint satisfaction at inference time.

### 3.2 Hard Constraints (Differentiable Repair Layers)

A differentiable repair layer is incorporated directly after the neural backbone, and gradients backpropagate through it. Existing architectures include OptNet (Amos & Kolter, 2021), DC3 (Donti et al., 2021), HardNet (Min & Azizan, 2025), and SnareNet (Chu et al., 2026).

Since this project targets differentiable layers in the **most general setting**, we focus on:

- **cvxpy layers** — for general convex optimization problems.
- **Theseus layers** — for general nonlinear differentiable optimization problems.

These general layers subsume specific repair layers (see Appendix). This choice extends the scope of the experiments considerably, as cvxpy and Theseus layers are widely used across many applications. The tradeoff is that direct task-specific implementations are faster to run.

---

## 4. Experimental Framework

### 4.1 Neural Network Backbones

Constraint enforcement techniques can be incorporated into any neural network architecture, making the study of training dynamics potentially dependent on the chosen backbone. To reach robust conclusions, the project investigates different backbones:

- **Current:** MLP architectures with various widths and depths.
- **Planned:** Transformer architectures.

### 4.2 Adaptive Relaxation

A significant finding from Chu et al. (2026) is the use of **adaptive relaxation** in constraint formulations to improve the learning abilities of hard-constrained methods. This project investigates the benefits of this technique through the lens of training dynamics.

### 4.3 Datasets

- **Current:** Synthetic randomly generated **QC-QP** (Quadratically Constrained Quadratic Program) problems, recovered from the experimental code of Chu et al. (2026). All three methods (soft, cvxpy, Theseus) can be applied to these.
- **Planned:** More practical settings such as basic robotics control problems or optimal powerflow instances.

---

## 5. Training Dynamics Metrics

See [docs/METRICS.md](docs/METRICS.md) for the full list of tracked metrics.

---

## 6. Preliminary Results & Current Status

- Significant work has been done to refine the project scope and set up the experimental plan around comparing soft penalty methods against cvxpy and Theseus layers.
- Existing repair strategies (e.g., SnareNet) have been incorporated within the general framework (see Appendix equivalence proof).
- A detailed literature review identified the most relevant metrics for investigating learning dynamics.
- A visual platform design has been proposed (inspired by LossLens).
- The codebase has been nearly completely set up.
- Experiments have started using basic MLP backbone architectures.

---

## 7. Remaining Work

1. **Fix cvxpy layers:** Previous cvxpy code was discovered to be incorrect and needs to be updated.
2. **Add loss landscape visualizations** based on Li et al. (2018).
3. Launch the most relevant experiments and iterate until satisfactory results are obtained.
4. Develop the final visual platform to display all experiment results together.

---

## Appendix: SnareNet as a Special Case of Theseus

### A.1 SnareNet Repair Layer

Given a neural model output $\hat{y}$, the repair layer $R$ modifies it to satisfy constraints by solving the nonlinear equation:

$$g(y) = z$$

where $z = P_{\mathcal{B}(l, u)}(g(\hat{y}))$ is the box-projection of $g(\hat{y})$ onto the constraint set $\mathcal{C}_x$.

This is solved using **Newton's method with Levenberg-Marquardt regularization**, giving the update rule for iterate $y^k$:

$$y^{k+1} = y^k - \underbrace{(J_g(y^k)^\top J_g(y^k) + \lambda I)^{-1}}_{J_\lambda^\dagger(y^k)} J_g(y^k)^\top (g(y^k) - z^k)$$

where $z^k = P_{\mathcal{B}(l, u)}(g(y^k))$ is the target image point, $J_g$ is the Jacobian of the constraint function $g$, and $\lambda \|y - y^k\|^2$ is a regularization term ensuring the update remains local.

*Note: Adaptive relaxation is omitted from this analysis as it does not change the equivalence.*

### A.2 Theseus DNLS Layer

The Theseus library defines a **Differentiable Nonlinear Least Squares (DNLS)** layer. Given input $\theta$, cost functions $c_i$, and cost weights $w_i$, Theseus minimizes:

$$S(\theta) = \frac{1}{2} \sum_i \|w_i \, c_i(\theta_i)\|^2$$

where $\theta_i$ denotes the subset of variables on which the $i$-th constraint depends. (This sparsity detail enables leveraging sparse solvers when constraints depend on small subsets of the input.)

### A.3 Equivalence Proof

To match SnareNet's formulation, adapt the Theseus layer as follows:

1. **Input:** Set $\theta = \hat{y}$ (ignore sparsity, drop the $i$ index).
2. **Cost functions:** Define $c_i(y) = g_i(y) - z_i$ (constraint violations at every index).
3. **Weights:** Set $w_i = 1$.

The Theseus layer then becomes:

$$R_{\text{Theseus}} : \hat{y} \mapsto \arg\min_y \frac{1}{2} \sum_i \|g_i(y) - z_i\|^2 =: \check{y}$$

Its internal Levenberg-Marquardt optimizer computes an update $\Delta$ by solving the linearized system:

$$(J^\top J + \lambda I) \Delta = J^\top r$$

where $r = g(y) - z$ is the residual vector and $J$ is the Jacobian of the residuals. **This is mathematically equivalent to the SnareNet update step (Eq. 4).**

**Important implementation detail:** The Theseus layer is only equivalent to the SnareNet repair step if the cost function is updated at every iteration to account for the update of $z^k$. This can be implemented by providing an `end_iter_callback` to the Theseus optimizer.


