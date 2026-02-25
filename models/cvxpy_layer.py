"""
CVXPy-layer model for QCQP optimisation learning.

Architecture:  MLP backbone  →  differentiable QCQP solve (cvxpylayers)

The backbone MLP predicts an objective vector (the linear cost term) that is
fed as a *parameter* into a small convex optimisation problem whose
constraints are the QCQP constraints.  The cvxpylayers library differentiates
through the solution via implicit differentiation of the KKT conditions.

The output is **always feasible** by construction.

Requires:  pip install cvxpylayers
"""

import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np

torch.set_default_dtype(torch.float64)

try:
    from cvxpylayers.torch import CvxpyLayer
    CVXPYLAYERS_AVAILABLE = True
except ImportError:
    CVXPYLAYERS_AVAILABLE = False

from .backbone import MLPBackbone


class CvxpyLayerNet(nn.Module):
    """
    MLP + differentiable QCQP projection via cvxpylayers.

    The backbone MLP maps x → p_pred (a learned cost vector).  Then a
    cvxpylayers solve computes:

        y* = argmin_y  0.5 y^T Q y + p_pred^T y
        s.t.  y^T H_i y + G_i^T y ≤ h_i,  i=1..m
              A y = x

    Gradients of y* w.r.t. the backbone parameters flow through the KKT
    implicit differentiation implemented by cvxpylayers.

    Parameters
    ----------
    problem     : QCQPProblem instance – supplies Q, A, G, H, h
    hidden_dim  : MLP hidden-layer width
    n_hidden    : number of hidden layers
    dropout     : dropout probability
    use_batchnorm : whether to use BatchNorm1d
    solver_args : extra kwargs forwarded to CvxpyLayer.forward (e.g.
                  solver_args={'eps': 1e-5, 'max_iters': 10000})
    """

    def __init__(
        self,
        problem,
        hidden_dim: int = 200,
        n_hidden: int = 2,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        solver_args: dict | None = None,
    ):
        if not CVXPYLAYERS_AVAILABLE:
            raise ImportError(
                "cvxpylayers is required for CvxpyLayerNet. "
                "Install with:  pip install cvxpylayers"
            )

        super().__init__()

        self._problem = problem
        self._solver_args = solver_args or {"eps": 1e-5, "max_iters": 10_000}

        n = problem.ydim       # number of decision variables
        e = problem.neq        # number of equality constraints
        m = problem.nineq      # number of inequality constraints

        # ---- Backbone MLP: x (dim e) → p_pred (dim n) ----
        self.backbone = MLPBackbone(
            input_dim=e,
            output_dim=n,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

        # ---- Build the cvxpylayers differentiable layer ----
        self._cvxpy_layer = self._build_cvxpy_layer(problem)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cvxpy_layer(problem) -> "CvxpyLayer":
        """
        Construct a CvxpyLayer for the QCQP.

        Parameters (fed at runtime):
            p_pred : (n,)  learned linear cost term
            x_param: (e,)  equality right-hand side

        Fixed data (baked into the layer):
            Q, A, G, H, h
        """
        n = problem.ydim
        e = problem.neq
        m = problem.nineq

        # Numpy copies of fixed problem data
        Q_np = problem.Q.cpu().numpy()
        A_np = problem.A.cpu().numpy()
        G_np = problem.G.cpu().numpy()
        H_np = problem.H.cpu().numpy()
        h_np = problem.h.cpu().numpy()

        # CVXPY variable
        y = cp.Variable(n)

        # CVXPY parameters (will be set at runtime per-sample)
        p_pred_param = cp.Parameter(n)       # learned cost vector
        x_param      = cp.Parameter(e)       # equality RHS

        # Objective: 0.5 y^T Q y + p_pred^T y
        objective = cp.Minimize(0.5 * cp.quad_form(y, Q_np) + p_pred_param @ y)

        # Constraints
        constraints = [A_np @ y == x_param]
        for i in range(m):
            constraints.append(
                cp.quad_form(y, H_np[i]) + G_np[i] @ y <= float(h_np[i])
            )

        prob = cp.Problem(objective, constraints)

        layer = CvxpyLayer(
            prob,
            parameters=[p_pred_param, x_param],
            variables=[y],
        )
        return layer

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
        y : (b, n)  feasible solution
        """
        # Backbone predicts a cost vector for each sample
        p_pred = self.backbone(x)  # (b, n)

        # Solve the QCQP for each sample via cvxpylayers
        (y_sol,) = self._cvxpy_layer(p_pred, x, solver_args=self._solver_args)

        return y_sol
