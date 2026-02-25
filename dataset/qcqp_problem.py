"""
QCQP problem definition for training dynamics comparison experiments.

Represents the family of convex Quadratically Constrained Quadratic Programs:

    minimize_y   0.5 * y^T Q y + p^T y
    subject to   y^T H_i y + G_i^T y  <=  h_i,   i = 1, ..., m   (inequality)
                 A y = x                                            (equality, parameterised by x)

For each problem instance x is drawn i.i.d. from a fixed distribution; the
goal is to train a neural network that maps x -> y*(x) directly.

All matrix/vector problem data (Q, p, A, G, H, h) are fixed at dataset
creation time. Only the right-hand side x varies across instances.

This file is self-contained: it only requires PyTorch, NumPy, cvxpy, and
standard library modules.
"""

import torch
import numpy as np
import cvxpy as cp
import time
import pickle
from tqdm import tqdm

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Dataset split helpers
# ---------------------------------------------------------------------------

def _make_split_indices(n_total: int, valid_frac: float, test_frac: float):
    train_end  = int(n_total * (1.0 - valid_frac - test_frac))
    valid_end  = int(n_total * (1.0 - test_frac))
    return (
        np.arange(0,          train_end),
        np.arange(train_end,  valid_end),
        np.arange(valid_end,  n_total),
    )


# ---------------------------------------------------------------------------
# Main problem class
# ---------------------------------------------------------------------------

class QCQPProblem:
    """
    Container for one QCQP family.  Holds problem data, all x-samples, optimal
    solutions and optimal values (when available).

    Parameters
    ----------
    Q   : (n, n)   ndarray  – PSD cost Hessian (diagonal in the generator)
    p   : (n,)     ndarray  – cost linear term
    A   : (e, n)   ndarray  – equality constraint matrix
    X   : (N, e)   ndarray  – all input samples
    G   : (m, n)   ndarray  – linear part of inequality constraints
    H   : (m, n, n) ndarray – quadratic part of inequality constraints
    h   : (m,)     ndarray  – right-hand side of inequalities
    L   : (n,)     ndarray  – variable lower bounds (used by solvers)
    U   : (n,)     ndarray  – variable upper bounds (used by solvers)
    valid_frac, test_frac   – dataset split fractions
    """

    def __init__(
        self,
        Q: np.ndarray,
        p: np.ndarray,
        A: np.ndarray,
        X: np.ndarray,
        G: np.ndarray,
        H: np.ndarray,
        h: np.ndarray,
        L: np.ndarray,
        U: np.ndarray,
        valid_frac: float = 0.0833,
        test_frac:  float = 0.0833,
    ):
        # Store problem data as double-precision tensors
        self._Q = torch.tensor(Q, dtype=torch.float64)   # (n, n)
        self._p = torch.tensor(p, dtype=torch.float64)   # (n,)
        self._A = torch.tensor(A, dtype=torch.float64)   # (e, n)
        self._G = torch.tensor(G, dtype=torch.float64)   # (m, n)
        self._H = torch.tensor(H, dtype=torch.float64)   # (m, n, n)
        self._h = torch.tensor(h, dtype=torch.float64)   # (m,)
        self._X = torch.tensor(X, dtype=torch.float64)   # (N, e)
        self._L = torch.tensor(L, dtype=torch.float64)   # (n,)
        self._U = torch.tensor(U, dtype=torch.float64)   # (n,)

        self._ydim   = Q.shape[0]
        self._neq    = A.shape[0]
        self._nineq  = h.shape[0]
        self._num    = X.shape[0]
        self._encoded_xdim = X.shape[1]
        self._valid_frac = valid_frac
        self._test_frac  = test_frac
        self._device = self._Q.device

        # Optimal solutions / values – filled by opt_solve()
        self._Y        = torch.full((self._num, self._ydim), float("nan"), dtype=torch.float64)
        self._opt_vals = torch.full((self._num,),            float("nan"), dtype=torch.float64)

        # Precompute train/valid/test index splits
        self._train_idx, self._valid_idx, self._test_idx = _make_split_indices(
            self._num, valid_frac, test_frac
        )


    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device):
        """Move all tensors to *device* in-place; returns self."""
        self._device = device
        for attr in ("_Q", "_p", "_A", "_G", "_H", "_h", "_L", "_U", "_X", "_Y", "_opt_vals"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ------------------------------------------------------------------
    # Properties – problem data
    # ------------------------------------------------------------------

    @property
    def Q(self):         return self._Q
    @property
    def p(self):         return self._p
    @property
    def A(self):         return self._A
    @property
    def G(self):         return self._G
    @property
    def H(self):         return self._H
    @property
    def h(self):         return self._h
    @property
    def ydim(self):      return self._ydim
    @property
    def neq(self):       return self._neq
    @property
    def nineq(self):     return self._nineq
    @property
    def num(self):       return self._num
    @property
    def encoded_xdim(self): return self._encoded_xdim
    @property
    def device(self):    return self._device
    @property
    def Y(self):         return self._Y
    @property
    def opt_vals(self):  return self._opt_vals


    # ------------------------------------------------------------------
    # Properties – dataset splits
    # ------------------------------------------------------------------

    @property
    def trainX(self):   return self._X[self._train_idx]
    @property
    def validX(self):   return self._X[self._valid_idx]
    @property
    def testX(self):    return self._X[self._test_idx]
    @property
    def trainY(self):   return self._Y[self._train_idx]
    @property
    def validY(self):   return self._Y[self._valid_idx]
    @property
    def testY(self):    return self._Y[self._test_idx]
    @property
    def trainOptvals(self): return self._opt_vals[self._train_idx]
    @property
    def validOptvals(self): return self._opt_vals[self._valid_idx]
    @property
    def testOptvals(self):  return self._opt_vals[self._test_idx]

    # ------------------------------------------------------------------
    # Objective and constraint residuals
    # ------------------------------------------------------------------

    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Objective value: 0.5 y^T Q y + p^T y. Shape: (b,)."""
        return (0.5 * (Y @ self.Q) * Y + self.p * Y).sum(dim=1)

    def get_ineq_res(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        ReLU(y^T H_i y + G_i^T y - h_i) for each inequality constraint.
        Returns zero when the constraint is satisfied.  Shape: (b, m).
        """
        quadratic = torch.einsum("bj,ijk,bk->bi", Y, self.H, Y)  # (b, m)
        linear    = torch.einsum("bj,ij->bi",     Y, self.G)      # (b, m)
        return torch.clamp(quadratic + linear - self.h, min=0.0)  # (b, m)

    def get_eq_res(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Absolute equality violation |A y - x|.  Shape: (b, e).
        """
        return torch.abs(Y @ self.A.T - X)  # (b, e)

    def get_resid(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Concatenation of inequality and equality residuals.  Shape: (b, m + e).
        """
        return torch.cat([self.get_ineq_res(X, Y), self.get_eq_res(X, Y)], dim=1)

    def get_lower_bound(self, X: torch.Tensor) -> torch.Tensor:
        """
        Lower bounds for all constraints in the format expected by projection
        models: [ -inf (for ineq) | x (for eq) ].  Shape: (b, m + e).
        """
        b = X.shape[0]
        bl_ineq = torch.full((b, self._nineq), -float("inf"), device=self.device)
        return torch.cat([bl_ineq, X], dim=1)

    def get_upper_bound(self, X: torch.Tensor) -> torch.Tensor:
        """
        Upper bounds for all constraints: [ h (for ineq) | x (for eq) ].
        Shape: (b, m + e).
        """
        bu_ineq = self.h.expand(X.shape[0], -1)
        return torch.cat([bu_ineq, X], dim=1)

    # ------------------------------------------------------------------
    # Loss functions (used by trainers for each comparison approach)
    # ------------------------------------------------------------------

    def get_objective_loss(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Pure objective loss (no constraint term). Shape: (b,).
        Used by hard-constraint methods (CVXPy layers, Theseus) where
        feasibility is enforced by the architecture.
        """
        return self.evaluate(X, Y)

    def get_soft_penalty_loss(
        self,
        Y: torch.Tensor,
        X: torch.Tensor,
        penalty_weight: float,
    ) -> torch.Tensor:
        """
        Soft-penalty loss:  objective  +  λ · ||residuals||².
        Shape: (b,).

        Parameters
        ----------
        Y              : predicted solutions, shape (b, n)
        X              : problem parameters,  shape (b, e)
        penalty_weight : λ – trades off feasibility vs. objective quality
        """
        obj = self.evaluate(X, Y)                               # (b,)
        viol = torch.norm(self.get_resid(X, Y), dim=1) ** 2    # (b,)
        return obj + penalty_weight * viol

    # ------------------------------------------------------------------
    # Constraint function interface (used by Theseus projection layer)
    # ------------------------------------------------------------------

    def g(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all constraint functions g(y).

        Returns the *raw* constraint values (not clamped), stacking
        inequality and equality constraints:
            g(y) = [ y^T H_i y + G_i^T y  |  A y ]
        Shape: (b, m + e).
        """
        ineq_g = (
            torch.einsum("bj,ijk,bk->bi", Y, self.H, Y)
            + torch.einsum("bj,ij->bi", Y, self.G)
        )  # (b, m)
        eq_g = torch.einsum("bj,ij->bi", Y, self.A)  # (b, e)
        return torch.cat([ineq_g, eq_g], dim=1)  # (b, m + e)

    def jacobian(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Jacobian of g(y) w.r.t. y.

        Returns dg/dy of shape (b, m + e, n).
        """
        # Inequality: d/dy [y^T H_i y + G_i^T y] = 2 H_i y + G_i
        J_ineq = (
            2 * torch.einsum("bk,ijk->bij", Y, self.H)
            + self.G.unsqueeze(0).expand(Y.shape[0], -1, -1)
        )  # (b, m, n)
        # Equality: d/dy [A y] = A
        J_eq = self.A.unsqueeze(0).expand(Y.shape[0], -1, -1)  # (b, e, n)
        return torch.cat([J_ineq, J_eq], dim=1)  # (b, m + e, n)

    def get_g(self, X: torch.Tensor):
        """Return the constraint function g (callable on Y)."""
        return self.g

    def get_jacobian(self, X: torch.Tensor):
        """Return the constraint Jacobian function (callable on Y)."""
        return self.jacobian

    # ------------------------------------------------------------------
    # CVXPY optimal solver
    # ------------------------------------------------------------------

    def opt_solve(self, indices=None, tol: float = 1e-5, verbose: bool = True) -> float:
        """
        Solve selected problem instances with CVXPY and store optimal solutions
        and objective values in self._Y and self._opt_vals.

        Parameters
        ----------
        indices : array-like of ints, or None (solve all)
        tol     : solver tolerance
        verbose : print progress bar

        Returns
        -------
        total_time : float  – wall-clock time in seconds
        """
        Q  = self._Q.cpu().numpy()
        p  = self._p.cpu().numpy()
        A  = self._A.cpu().numpy()
        G  = self._G.cpu().numpy()
        H  = self._H.cpu().numpy()
        h  = self._h.cpu().numpy()
        X_all = self._X.cpu().numpy()

        if indices is None:
            indices = np.arange(self._num)

        total_time = 0.0
        it = tqdm(indices, desc="Solving QCQP (cvxpy)") if verbose else indices

        for idx in it:
            x_i = X_all[idx]
            y   = cp.Variable(self._ydim)

            constraints = [A @ y == x_i]
            for i in range(self._nineq):
                constraints.append(cp.quad_form(y, H[i]) + G[i] @ y <= h[i])

            objective = cp.Minimize(0.5 * cp.quad_form(y, Q) + p @ y)
            prob = cp.Problem(objective, constraints)

            t0 = time.time()
            prob.solve(solver=cp.SCS, eps=tol)
            total_time += time.time() - t0

            if prob.status in ("optimal", "optimal_inaccurate") and y.value is not None:
                self._Y[idx]        = torch.tensor(y.value, dtype=torch.float64)
                self._opt_vals[idx] = float(prob.value)

        return total_time

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Serialise the dataset to *path* using pickle."""
        payload = {
            "Q": self._Q.cpu().numpy(),
            "p": self._p.cpu().numpy(),
            "A": self._A.cpu().numpy(),
            "G": self._G.cpu().numpy(),
            "H": self._H.cpu().numpy(),
            "h": self._h.cpu().numpy(),
            "L": self._L.cpu().numpy(),
            "U": self._U.cpu().numpy(),
            "X": self._X.cpu().numpy(),
            "Y": self._Y.cpu().numpy(),
            "opt_vals":   self._opt_vals.cpu().numpy(),
            "valid_frac": self._valid_frac,
            "test_frac":  self._test_frac,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[QCQPProblem] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "QCQPProblem":
        """Load a dataset previously saved with :meth:`save`."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        instance = cls(
            Q=d["Q"], p=d["p"], A=d["A"], X=d["X"],
            G=d["G"], H=d["H"], h=d["h"],
            L=d["L"], U=d["U"],
            valid_frac=d.get("valid_frac", 0.0833),
            test_frac=d.get("test_frac",  0.0833),
        )
        instance._Y        = torch.tensor(d["Y"],        dtype=torch.float64)
        instance._opt_vals = torch.tensor(d["opt_vals"], dtype=torch.float64)
        return instance

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"QCQPProblem(n={self._ydim}, m_ineq={self._nineq}, "
            f"m_eq={self._neq}, N={self._num})"
        )
