"""
CVXPY-layer model for QCQP optimisation learning.

Architecture:  MLP / Transformer backbone  →  cvxpylayers projection

The backbone predicts an initial point ŷ = M_θ(x).  A cvxpylayers
differentiable projection layer then solves a Euclidean projection:

    ỹ = argmin_y  0.5 ‖y − ŷ‖²
         s.t.     A y  = x                          (equality)
                  y^T H_i y + G_i^T y  ≤  h_i + ε_i  (quadratic inequality)
                  L  ≤  y  ≤  U                      (variable bounds)

Gradients flow via implicit differentiation of the KKT conditions
(see CVXPY_CONTEXT.md for mathematical details).

Requires:  pip install cvxpylayers
"""

import warnings

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from cvxpylayers.torch import CvxpyLayer

from .backbone_factory import build_backbone


class CvxpyLayerNet(nn.Module):
    """
    Backbone + cvxpylayers Euclidean projection for hard QCQP constraints.

    Parameters
    ----------
    problem        : QCQPProblem instance — supplies constraint matrices
                     (Q, p, A, G, H, h, L, U) and problem dimensions.
    backbone_type  : 'mlp' or 'transformer'
    solver         : CVXPY solver to use (default: cp.SCS)
    solver_eps     : SCS solver tolerance (default: 1e-5)
    solver_max_iters : SCS maximum iterations (default: 10_000)
    **backbone_kwargs : extra arguments forwarded to the backbone constructor
    """

    def __init__(
        self,
        problem,
        backbone_type: str = "mlp",
        solver: str = "SCS",
        solver_eps: float = 1e-5,
        solver_max_iters: int = 10_000,
        **backbone_kwargs,
    ):
        super().__init__()

        self._problem = problem
        self._solver = solver
        self._solver_eps = solver_eps
        self._solver_max_iters = solver_max_iters

        n = problem.ydim
        e = problem.neq
        m = problem.nineq

        # Relaxation epsilon — can be set externally for adaptive relaxation
        self._eps = torch.zeros(m + e)

        # ---- Backbone: x (dim e) → ŷ (dim n) ----
        self.backbone = build_backbone(
            backbone_type,
            input_dim=e,
            output_dim=n,
            **backbone_kwargs,
        )

        # ---- Build cvxpylayers problem (once) ----
        self._build_cvxpy_layer(problem)

    # ------------------------------------------------------------------
    # Build the parametric CVXPY problem + CvxpyLayer
    # ------------------------------------------------------------------

    def _build_cvxpy_layer(self, problem):
        """Construct the parametric CVXPY projection problem.

        Parameters (change per instance):
          - y_hat  (n,)  : backbone prediction to project
          - x_param (e,) : equality constraint RHS
          - eps_param (m,) : per-constraint relaxation for inequality constraints

        Constants (fixed for the problem family):
          - A, G, H, h, L, U
        """
        n = problem.ydim
        e = problem.neq
        m = problem.nineq

        # Decision variable
        y = cp.Variable(n, name="y")

        # Parameters (batched by cvxpylayers)
        y_hat = cp.Parameter(n, name="y_hat")
        x_param = cp.Parameter(e, name="x_param")
        eps_ineq = cp.Parameter(m, name="eps_ineq", nonneg=True)

        # Constant problem data (numpy, float64)
        A_np = problem.A.cpu().numpy()
        G_np = problem.G.cpu().numpy()
        H_np = problem.H.cpu().numpy()  # (m, n, n)
        h_np = problem.h.cpu().numpy()
        L_np = problem._L.cpu().numpy()
        U_np = problem._U.cpu().numpy()

        # Objective: min 0.5 ||y - y_hat||^2
        objective = cp.Minimize(0.5 * cp.sum_squares(y - y_hat))

        # Constraints
        constraints = []

        # Equality: A y = x
        constraints.append(A_np @ y == x_param)

        # Quadratic inequality: y^T H_i y + G_i^T y <= h_i + eps_i
        for i in range(m):
            H_i = H_np[i]
            # Symmetrise H_i for numerical robustness
            H_i_sym = 0.5 * (H_i + H_i.T)
            constraints.append(
                cp.quad_form(y, H_i_sym) + G_np[i] @ y <= h_np[i] + eps_ineq[i]
            )

        # Variable bounds: L <= y <= U (only add finite bounds)
        if np.all(np.isfinite(L_np)):
            constraints.append(y >= L_np)
        elif np.any(np.isfinite(L_np)):
            # Add element-wise bounds for finite entries only
            for j in range(n):
                if np.isfinite(L_np[j]):
                    constraints.append(y[j] >= L_np[j])

        if np.all(np.isfinite(U_np)):
            constraints.append(y <= U_np)
        elif np.any(np.isfinite(U_np)):
            for j in range(n):
                if np.isfinite(U_np[j]):
                    constraints.append(y[j] <= U_np[j])

        prob = cp.Problem(objective, constraints)

        # Verify DCP compliance
        if not prob.is_dcp():
            raise ValueError(
                "CVXPY projection problem is not DCP-compliant. "
                "Check that all H_i matrices are PSD (convex QCQP)."
            )

        # Wrap as a differentiable layer
        self._cvxpy_layer = CvxpyLayer(
            prob,
            parameters=[y_hat, x_param, eps_ineq],
            variables=[y],
        )

        # Store references for introspection
        self._cvxpy_prob = prob
        self._cvxpy_var = y

    # ------------------------------------------------------------------
    # Epsilon control (for adaptive relaxation experiments)
    # ------------------------------------------------------------------

    def set_eps(self, eps: torch.Tensor):
        """Set constraint relaxation.  eps should have shape (m + e,)."""
        self._eps = eps

    def get_eps(self) -> torch.Tensor:
        return self._eps

    # ------------------------------------------------------------------
    # Solver kwargs
    # ------------------------------------------------------------------

    def _solver_kwargs(self) -> dict:
        """Build solver keyword arguments for the CvxpyLayer forward pass."""
        kwargs = {}
        solver_name = self._solver.upper()
        if solver_name == "SCS":
            kwargs["solver_args"] = {
                "eps": self._solver_eps,
                "max_iters": self._solver_max_iters,
                "verbose": False,
            }
        elif solver_name == "ECOS":
            kwargs["solver_args"] = {
                "abstol": self._solver_eps,
                "reltol": self._solver_eps,
                "max_iters": self._solver_max_iters,
                "verbose": False,
            }
        return kwargs

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def _project(self, y0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project backbone output y0 onto the feasible set C(x).

        Parameters
        ----------
        y0 : (b, n)  backbone predictions
        x  : (b, e)  equality constraint right-hand sides

        Returns
        -------
        y  : (b, n)  projected (feasible) predictions
        """
        b = y0.shape[0]
        m = self._problem.nineq

        # Inequality relaxation (only the first m entries matter for ineq)
        eps = self._eps[:m].to(y0.device)
        eps_batch = eps.unsqueeze(0).expand(b, -1)  # (b, m)

        try:
            (y_proj,) = self._cvxpy_layer(
                y0, x, eps_batch,
                **self._solver_kwargs(),
            )
        except Exception as e:
            warnings.warn(
                f"cvxpylayers solve failed: {e}. "
                f"Returning backbone prediction (infeasible).",
                stacklevel=2,
            )
            return y0

        return y_proj

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
        y : (b, n)  feasible solution (projection of backbone output)
        """
        y0 = self.backbone(x)
        return self._project(y0, x)
