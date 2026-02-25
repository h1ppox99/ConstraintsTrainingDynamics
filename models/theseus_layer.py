"""
Theseus-layer model for QCQP optimisation learning.

Architecture:  MLP backbone  →  Theseus projection (nonlinear least-squares)

The backbone MLP predicts an initial point y₀.  A Theseus differentiable
nonlinear least-squares solver then projects y₀ onto the feasible set by
driving constraint violations to zero.

The constraint violation is formulated as a cost function:
    F(y) = -(ReLU(bl - ε - g(y)) - ReLU(g(y) - bu - ε))
where g(y) stacks the quadratic inequality values and the linear equality
values.  The Theseus LM/GN solver minimises ||F(y)||² starting from y₀.

Gradients flow through the solver via one of several backward modes:
  - UNROLL  : backprop through all inner iterations (default, matches Newton)
  - IMPLICIT: implicit differentiation at the fixed point
  - TRUNCATED / DLM : truncated unrolling variants

Requires:  pip install theseus-ai
"""

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from typing import Optional, Tuple, List

import theseus as th
from theseus.core import CostFunction, Objective, Variable

from .backbone import MLPBackbone


# ---------------------------------------------------------------------------
# Helpers for the Theseus cost function
# ---------------------------------------------------------------------------

class _FuncHolder:
    """Mutable container for constraint functions that change per forward pass."""
    __slots__ = ("g_func", "J_func")

    def __init__(self):
        self.g_func = None
        self.J_func = None


class _ConstraintViolationCost(CostFunction):
    """
    Theseus CostFunction encoding constraint violation.

    error():
        F(y) = -(ReLU(bl - ε - g(y)) - ReLU(g(y) - bu - ε))
        → zero when all constraints are satisfied within relaxation ε.

    jacobians():
        Returns the full constraint Jacobian J_g = dg/dy (not dF/dy).
        This matches the Gauss-Newton / LM convention: the solver computes
            Δy = -(J_g^T J_g + λI)^{-1} J_g^T F(y)
    """

    def __init__(
        self,
        y_var: "th.Vector",
        bl_var: Variable,
        bu_var: Variable,
        eps_var: Variable,
        func_holder: _FuncHolder,
        n_constraints: int,
        cost_weight=None,
        name: Optional[str] = None,
    ):
        if cost_weight is None:
            cost_weight = th.ScaleCostWeight(1.0)
        super().__init__(cost_weight, name=name)

        self._func_holder = func_holder
        self._n_constraints = n_constraints

        self.y_var = y_var
        self.bl_var = bl_var
        self.bu_var = bu_var
        self.eps_var = eps_var

        self.register_vars([y_var], is_optim_vars=True)
        self.register_vars([bl_var, bu_var, eps_var], is_optim_vars=False)

    def error(self) -> torch.Tensor:
        g_y = self._func_holder.g_func(self.y_var.tensor)
        bl = self.bl_var.tensor
        bu = self.bu_var.tensor
        eps = self.eps_var.tensor
        return -(torch.relu(bl - eps - g_y) - torch.relu(g_y - bu - eps))

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        y = self.y_var.tensor
        J_g = self._func_holder.J_func(y)
        err = self.error()
        return [J_g], err

    def dim(self) -> int:
        return self._n_constraints

    def _copy_impl(self, new_name=None) -> "_ConstraintViolationCost":
        return _ConstraintViolationCost(
            self.y_var.copy(), self.bl_var.copy(),
            self.bu_var.copy(), self.eps_var.copy(),
            self._func_holder, self._n_constraints,
            cost_weight=self.weight.copy(), name=new_name,
        )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TheseusLayerNet(nn.Module):
    """
    MLP + Theseus nonlinear-least-squares projection for hard QCQP constraints.

    Parameters
    ----------
    problem          : QCQPProblem instance – supplies constraint functions and
                       bounds via get_g(), get_jacobian(), get_lower_bound(),
                       get_upper_bound().
    hidden_dim       : MLP hidden-layer width
    n_hidden         : number of hidden layers
    dropout          : dropout probability
    use_batchnorm    : whether to use BatchNorm1d
    newton_maxiter   : max inner-solver iterations        (default 50)
    rtol             : convergence tolerance               (default 1e-5)
    lambd            : LM damping / Tikhonov parameter     (default 1e-4)
    backward_mode    : 'unroll' | 'implicit' | 'truncated' | 'dlm'
    optimizer_type   : 'levenberg_marquardt' | 'gauss_newton'
    trust_region     : clip final delta to [-1, 1]         (default False)
    """

    def __init__(
        self,
        problem,
        hidden_dim: int = 200,
        n_hidden: int = 2,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        newton_maxiter: int = 50,
        rtol: float = 1e-5,
        lambd: float = 1e-4,
        backward_mode: str = "unroll",
        optimizer_type: str = "levenberg_marquardt",
        trust_region: bool = False,
    ):
        super().__init__()

        self._problem = problem
        self._newton_maxiter = newton_maxiter
        self._rtol = rtol
        self._lambd = lambd
        self._backward_mode = backward_mode.lower()
        self._optimizer_type = optimizer_type.lower()
        self._trust_region = trust_region

        n = problem.ydim
        e = problem.neq
        m = problem.nineq
        self._n_constraints = m + e

        # Relaxation epsilon – can be set externally for adaptive relaxation
        self._eps = torch.zeros(m + e)

        # ---- Backbone MLP: x (dim e) → y₀ (dim n) ----
        self.backbone = MLPBackbone(
            input_dim=e,
            output_dim=n,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        # ---- Theseus infrastructure (built lazily on first forward) ----
        self._func_holder = _FuncHolder()
        self.__dict__["_theseus_built"] = False

    # ------------------------------------------------------------------
    # Epsilon control (for adaptive relaxation experiments)
    # ------------------------------------------------------------------

    def set_eps(self, eps: torch.Tensor):
        self._eps = eps

    def get_eps(self) -> torch.Tensor:
        return self._eps

    # ------------------------------------------------------------------
    # Lazy Theseus build
    # ------------------------------------------------------------------

    def _build_theseus(self, device, dtype):
        """Build Objective + Optimizer once; reuse across forward passes."""
        n = self._problem.ydim
        nc = self._n_constraints

        y_var   = th.Vector(tensor=torch.zeros(1, n,  dtype=dtype, device=device), name="y")
        bl_var  = Variable(torch.zeros(1, nc, dtype=dtype, device=device), name="bl")
        bu_var  = Variable(torch.zeros(1, nc, dtype=dtype, device=device), name="bu")
        eps_var = Variable(torch.zeros(1, nc, dtype=dtype, device=device), name="eps")

        cost = _ConstraintViolationCost(
            y_var, bl_var, bu_var, eps_var,
            self._func_holder, nc, name="constraint_violation",
        )

        objective = Objective(dtype=dtype)
        objective.add(cost)
        objective.to(device=device)
        objective.update()

        damping = max(self._lambd, 1e-10)

        if "gauss" in self._optimizer_type:
            optimizer = th.GaussNewton(
                objective, vectorize=True,
                max_iterations=self._newton_maxiter,
                abs_err_tolerance=self._rtol ** 2,
                rel_err_tolerance=self._rtol,
                step_size=1.0,
            )
        else:
            optimizer = th.LevenbergMarquardt(
                objective, vectorize=True,
                max_iterations=self._newton_maxiter,
                abs_err_tolerance=self._rtol ** 2,
                rel_err_tolerance=self._rtol,
                step_size=1.0,
            )

        layer = th.TheseusLayer(optimizer, vectorize=True)
        layer.to(device=device, dtype=dtype)

        opt_kwargs = {
            "track_best_solution": False,
            "track_err_history": False,
            "track_state_history": False,
        }
        if "gauss" not in self._optimizer_type:
            opt_kwargs["damping"] = damping
            opt_kwargs["adaptive_damping"] = False

        self.__dict__["_theseus_layer"]     = layer
        self.__dict__["_theseus_objective"] = objective
        self.__dict__["_theseus_optimizer"] = optimizer
        self.__dict__["_opt_kwargs"]        = opt_kwargs
        self.__dict__["_theseus_built"]     = True

    # ------------------------------------------------------------------
    # Backward mode resolution
    # ------------------------------------------------------------------

    def _resolve_backward_mode(self):
        import warnings
        mode_map = {
            "unroll":    th.BackwardMode.UNROLL,
            "implicit":  th.BackwardMode.IMPLICIT,
            "truncated": th.BackwardMode.TRUNCATED,
            "dlm":       th.BackwardMode.DLM,
        }
        mode = self._backward_mode
        if mode not in mode_map:
            raise ValueError(f"Unknown backward_mode '{mode}'. Choose from {list(mode_map)}.")
        if mode != "unroll" and self.training:
            warnings.warn(
                f"backward_mode='{mode}' does not propagate gradients through "
                "the backbone MLP → solver input.  Falling back to 'unroll' "
                "during training.", stacklevel=2,
            )
            return th.BackwardMode.UNROLL
        return mode_map[mode]

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _project(self, y0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project y0 onto the feasible set using the Theseus solver."""
        bl = self._problem.get_lower_bound(x)
        bu = self._problem.get_upper_bound(x)
        g_func = self._problem.get_g(x)
        J_func = self._problem.get_jacobian(x)

        self._func_holder.g_func = g_func
        self._func_holder.J_func = J_func

        if not self.__dict__["_theseus_built"]:
            self._build_theseus(y0.device, y0.dtype)

        b = y0.shape[0]
        eps_tensor = self._eps.to(y0.device).unsqueeze(0).expand(b, -1)

        backward_mode = self._resolve_backward_mode()

        input_tensors = {"y": y0, "bl": bl, "bu": bu, "eps": eps_tensor}
        opt_kwargs = {"backward_mode": backward_mode, **self.__dict__["_opt_kwargs"]}

        result, _ = self.__dict__["_theseus_layer"].forward(
            input_tensors=input_tensors, optimizer_kwargs=opt_kwargs,
        )
        y_result = result["y"]

        if self._trust_region:
            delta = torch.clamp(y_result - y0, -1, 1)
            y_result = y0 + delta

        return y_result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (b, e)  problem parameters (equality RHS)

        Returns
        -------
        y : (b, n)  (approximately) feasible solution
        """
        y0 = self.backbone(x)
        return self._project(y0, x)
