"""
Suite of 5 progressively harder QCQP datasets.

All datasets are generated using the same ``generate()`` function from
``generate_dataset.py`` but with different arguments controlling problem
size and number of instances.  Every dataset is solved with CVXPY (SCS)
and saved to ``dataset/data/``.

Problem formulation (same for all datasets):
    minimize_y   0.5 * y^T Q y + p^T y
    subject to   y^T H_i y + G_i^T y  <=  h_i,   i = 1, ..., m
                 A y = x

Difficulty is increased along three axes:
  - num_var  (n)  : decision variables
  - num_ineq (m)  : inequality constraints
  - num_eq   (e)  : equality constraints  (= dimension of the parameter x)
  - n_examples(N) : number of problem instances

Baseline (existing dataset):
    n=50,  m=20,  e=20,  N=2_000,  seed=2025

Dataset configurations
----------------------
  D1  – slightly harder   : n=75,  m=35,  e=35,  N=3_000,  seed=42
  D2  – moderate          : n=100, m=50,  e=50,  N=5_000,  seed=123
  D3  – medium-hard       : n=150, m=75,  e=60,  N=5_000,  seed=456
  D4  – hard              : n=200, m=100, e=80,  N=4_000,  seed=789
  D5  – very hard         : n=300, m=150, e=100, N=3_000,  seed=1337

Usage
-----
    # From the repository root:
    python -m dataset.generate_datasets_suite

    # or directly:
    python dataset/generate_datasets_suite.py
"""

import os
import sys
import time

# Make sure repository root is on the path when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset.generate_dataset import generate  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASETS = [
    # (label,           num_var, num_ineq, num_eq, n_examples, seed)
    ("D1 – slightly harder",  75,   35,   35,  3_000,   42),
    ("D2 – moderate",        100,   50,   50,  5_000,  123),
    ("D3 – medium-hard",     150,   75,   60,  5_000,  456),
    ("D4 – hard",            200,  100,   80,  4_000,  789),
    ("D5 – very hard",       300,  150,  100,  3_000, 1337),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_wall = 0.0
    results = []

    for label, num_var, num_ineq, num_eq, n_examples, seed in DATASETS:
        print(f"\n{'#'*70}")
        print(f"  Generating: {label}")
        print(f"  n={num_var}, m_ineq={num_ineq}, m_eq={num_eq}, "
              f"N={n_examples}, seed={seed}")
        print(f"{'#'*70}")

        t0 = time.time()
        problem = generate(
            num_var    = num_var,
            num_ineq   = num_ineq,
            num_eq     = num_eq,
            n_examples = n_examples,
            seed       = seed,
            opt_solve  = True,          # always solve with CVXPY
            output_dir = OUTPUT_DIR,
            cvxpy_tol  = 1e-5,
        )
        elapsed = time.time() - t0
        total_wall += elapsed

        results.append((label, num_var, num_ineq, num_eq, n_examples, seed, elapsed))
        print(f"  Wall time: {elapsed:.1f}s  ({elapsed/n_examples*1000:.1f} ms/instance)")

    # Summary table
    print(f"\n{'='*70}")
    print("  Generation Summary")
    print(f"{'='*70}")
    print(f"  {'Dataset':<25} {'n':>5} {'m':>5} {'e':>5} {'N':>7}  {'Time':>10}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*7}  {'-'*10}")
    for label, n, m, e, N, seed, elapsed in results:
        print(f"  {label:<25} {n:>5} {m:>5} {e:>5} {N:>7}  {elapsed:>9.1f}s")
    print(f"  {'TOTAL':<25} {'':>5} {'':>5} {'':>5} {'':>7}  {total_wall:>9.1f}s")
    print(f"{'='*70}\n")

    print(f"All datasets saved to:  {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main()
