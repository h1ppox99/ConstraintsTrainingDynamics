"""
Dataset generator for the QCQP training-dynamics comparison experiments.

Generates a family of convex Quadratically Constrained Quadratic Programs:

    minimize_y   0.5 * y^T Q y + p^T y
    subject to   y^T H_i y + G_i^T y  <=  h_i,   i = 1, ..., m
                 A y = x

Problem matrices (Q, p, A, G, H, h) are drawn once from fixed random
distributions; only the right-hand side x varies across the N instances.
The inequality upper bounds h are computed automatically so that the
pseudo-inverse solution y = A^+ x is always feasible (with a safety margin).

Usage
-----
# Minimal – generate and solve 10 000 instances (100 vars, 50 ineq, 50 eq):
python generate_dataset.py

# Custom problem size:
python generate_dataset.py --num_var 50 --num_ineq 20 --num_eq 20 --n_examples 5000

# Skip the CVXPY solve (no optimal labels):
python generate_dataset.py --no_opt_solve

# Custom output directory and seed:
python generate_dataset.py --output_dir ./data --seed 42

The script saves one file:
    <output_dir>/qcqp_var{v}_ineq{m}_eq{e}_N{n}_seed{s}.pkl
"""

import argparse
import os
import numpy as np

from dataset.qcqp_problem import QCQPProblem


# ---------------------------------------------------------------------------
# Feasible bound computation
# ---------------------------------------------------------------------------

def compute_feasible_ineq_bounds(
    A: np.ndarray,
    H: np.ndarray,
    G: np.ndarray,
    X: np.ndarray,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Compute h such that the pseudo-inverse solution y = A^+ x satisfies all
    inequality constraints for every x in X.

    For each constraint i, the worst-case value is bounded by:
        max_x  x^T (A^+)^T H_i A^+ x + G_i^T A^+ x
            <= ||H_i'||_2 * ||x||_max^2  +  ||G_i'||_2 * ||x||_max
    where  H_i' = (A^+)^T H_i A^+,  G_i' = G_i A^+.

    Parameters
    ----------
    A      : (e, n)    equality matrix
    H      : (m, n, n) quadratic inequality matrices
    G      : (m, n)    linear inequality matrix
    X      : (N, e)    input samples
    margin : additive safety margin

    Returns
    -------
    h : (m,) upper bounds ensuring feasibility of y = A^+ x for all x
    """
    Apinv       = np.linalg.pinv(A)                                         # (n, e)
    quad_mat    = np.einsum("va,ivw,wb->iab", Apinv, H, Apinv)             # (m, e, e)
    lin_vec     = G @ Apinv                                                  # (m, e)
    max_x_norm  = np.max(np.linalg.norm(X, axis=1))                         # scalar
    quad_norms  = np.linalg.norm(quad_mat, ord=2, axis=(1, 2))              # (m,) spectral norms
    lin_norms   = np.linalg.norm(lin_vec, axis=1)                           # (m,)
    h = quad_norms * max_x_norm ** 2 + lin_norms * max_x_norm + margin      # (m,)
    return h


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def generate(
    num_var:    int   = 100,
    num_ineq:   int   = 50,
    num_eq:     int   = 50,
    n_examples: int   = 10_000,
    seed:       int   = 2025,
    opt_solve:  bool  = True,
    output_dir: str   = ".",
    cvxpy_tol:  float = 1e-8,
) -> QCQPProblem:
    """
    Generate a QCQP dataset and (optionally) solve all instances with CVXPY.

    The random distributions follow the conventions of the parent repository:
    - Q  ~ Diag(Uniform[0, 0.5])           PSD diagonal cost Hessian
    - p  ~ Uniform[-1, 1]                  cost linear term
    - A  ~ Uniform[-1, 1]                  equality matrix
    - X  ~ Uniform[-0.5, 0.5]              input samples (RHS of equalities)
    - G  ~ Uniform[-1, 1]                  linear ineq. coefficients
    - H  ~ Diag(Uniform[0, 0.1])           quadratic ineq. matrices (PSD diagonal)
    - h  : computed via compute_feasible_ineq_bounds
    - L, U = ±10 (variable box bounds, used by external solvers only)

    Parameters
    ----------
    num_var    : number of decision variables n
    num_ineq   : number of inequality constraints m
    num_eq     : number of equality constraints e  (= dimension of x)
    n_examples : total number of instances N
    seed       : numpy random seed for reproducibility
    opt_solve  : solve all instances with CVXPY if True
    output_dir : where to save the .pkl file
    cvxpy_tol  : solver tolerance for CVXPY

    Returns
    -------
    problem : QCQPProblem  (already saved to disk)
    """
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"  QCQP Dataset Generation")
    print(f"  n={num_var},  m_ineq={num_ineq},  m_eq={num_eq},  N={n_examples},  seed={seed}")
    print(f"{'='*60}\n")

    # ---- Problem matrices ------------------------------------------------
    print("Sampling problem matrices...")
    Q = np.diag(rng.uniform(0.0, 0.5, num_var))              # (n, n) PSD diagonal
    p = rng.uniform(-1.0, 1.0, num_var)                       # (n,)
    A = rng.uniform(-1.0, 1.0, size=(num_eq, num_var))        # (e, n)
    X = rng.uniform(-0.5, 0.5, size=(n_examples, num_eq))     # (N, e)
    G = rng.uniform(-1.0, 1.0, size=(num_ineq, num_var))      # (m, n)
    H_diag = rng.uniform(0.0, 0.1, size=(num_ineq, num_var))  # (m, n)
    H = np.array([np.diag(H_diag[i]) for i in range(num_ineq)])  # (m, n, n) PSD diagonal
    L = np.full(num_var, -10.0)
    U = np.full(num_var,  10.0)

    # ---- Feasible inequality bounds -------------------------------------
    print("Computing feasibility-preserving inequality bounds h...")
    h = compute_feasible_ineq_bounds(A, H, G, X, margin=0.1)

    # ---- Build problem instance -----------------------------------------
    print("Building QCQPProblem instance...")
    problem = QCQPProblem(Q=Q, p=p, A=A, X=X, G=G, H=H, h=h, L=L, U=U)
    print(f"  {problem}")

    # ---- Optimal solve --------------------------------------------------
    if opt_solve:
        print(f"\nSolving {n_examples} instances with CVXPY (tol={cvxpy_tol})...")
        t = problem.opt_solve(tol=cvxpy_tol)
        print(f"  Done in {t:.1f}s  ({t/n_examples*1000:.1f} ms/instance)")
    else:
        print("\nSkipping CVXPY solve (--no_opt_solve flag set).")

    # ---- Save -----------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    filename = f"qcqp_var{num_var}_ineq{num_ineq}_eq{num_eq}_N{n_examples}_seed{seed}.pkl"
    save_path = os.path.join(output_dir, filename)
    problem.save(save_path)

    return problem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a random QCQP dataset for training-dynamics experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_var",    type=int,   default=100,    help="Number of decision variables n")
    parser.add_argument("--num_ineq",   type=int,   default=50,     help="Number of inequality constraints m")
    parser.add_argument("--num_eq",     type=int,   default=50,     help="Number of equality constraints e")
    parser.add_argument("--n_examples", type=int,   default=10_000, help="Total number of instances N")
    parser.add_argument("--seed",       type=int,   default=2025,   help="Random seed")
    parser.add_argument("--no_opt_solve", action="store_true",      help="Skip CVXPY solve (no optimal labels)")
    parser.add_argument("--output_dir", type=str,   default=".",    help="Directory to save the dataset file")
    parser.add_argument("--cvxpy_tol",  type=float, default=1e-5,   help="CVXPY solver tolerance")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        num_var    = args.num_var,
        num_ineq   = args.num_ineq,
        num_eq     = args.num_eq,
        n_examples = args.n_examples,
        seed       = args.seed,
        opt_solve  = not args.no_opt_solve,
        output_dir = args.output_dir,
        cvxpy_tol  = args.cvxpy_tol,
    )
