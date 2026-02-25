# Training Dynamics Comparison Experiments

## Problem setting

We consider a parametric family of convex Quadratically Constrained Quadratic
Programs (QCQPs):

$$\min_{y} \;\; \tfrac{1}{2} y^\top Q y + p^\top y$$

$$\text{subject to} \quad y^\top H_i y + G_i^\top y \leq h_i, \quad i = 1,\ldots,m$$

$$\hphantom{\text{subject to} \quad} A y = x$$

The problem data $(Q, p, A, G, H, h)$ are fixed; only the right-hand side
$x \in \mathbb{R}^e$ varies across instances.  A neural network is trained to
map $x \mapsto y^*(x)$ directly, bypassing the need to run an iterative solver
at test time.

The dataset is generated with `dataset/generate_dataset.py` and stored as a
single `.pkl` file.  See [`dataset/`](#folder-structure) for details.


---

## Constraint enforcement techniques

### 1. Soft Penalty

**Core idea.** Add a differentiable penalty for constraint violation directly
to the training loss:

$$\mathcal{L}_\lambda(\theta) = \mathbb{E}_x \bigl[ f(x, y_\theta(x)) + \lambda \cdot \|\text{res}(x, y_\theta(x))\|^2 \bigr]$$

where $\text{res}$ stacks the ReLU inequality residuals and the absolute
equality residuals, and $\lambda > 0$ is a scalar penalty weight.

**Architecture.** A plain MLP with no special output layer.  Feasibility is
encouraged through the loss, but never guaranteed.

**Key dynamics questions.**
- How does the choice of $\lambda$ affect the trade-off between objective
  quality and constraint satisfaction?
- Does increasing $\lambda$ over training (penalty annealing) lead to better
  convergence?
- Do gradients from the penalty term dominate or interfere with the objective
  gradient?

---

### 2. Theseus Layers (differentiable nonlinear optimizer)

**Core idea.** Embed a differentiable nonlinear least-squares solver
(Levenberg-Marquardt or Gauss-Newton) as a learnable layer.  The constraint
violation is formulated as a cost function minimised by the Theseus
optimizer; gradients flow through the iterative solve via unrolling,
implicit differentiation, or a Truncated / DLM mode.

**Architecture.** Backbone MLP (predicts an initial point) → *Theseus
projection layer* (Newton-style correction until feasible) → feasible output
$y$.  The output satisfies constraints up to solver tolerance.

**Differentiability.** Multiple backward modes are available:
- **Unroll** – standard backpropagation through the solver iterations (exact
  but memory-heavy for many iterations).
- **Implicit** – implicit differentiation via the fixed-point condition
  (memory-efficient but requires the solve to converge).
- **Truncated / DLM** – truncated unrolling or direct loss minimisation
  (intermediate memory/accuracy trade-off).

**Key dynamics questions.**
- How does the choice of backward mode affect gradient quality and training
  stability?
- How does the number of inner solver iterations interact with gradient signal
  for the backbone?
- Does the warm-start from the backbone MLP meaningfully speed up inner-loop
  convergence, and does this improve over training?

**References.** Pineda et al., "Theseus: A Library for Differentiable
Nonlinear Optimization" (NeurIPS 2022); `theseus-ai` library.


