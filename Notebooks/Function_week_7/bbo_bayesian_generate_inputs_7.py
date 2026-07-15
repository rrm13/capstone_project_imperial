#!/usr/bin/env python3
"""
Bayesian Optimization script for a Black Box Optimization (BBO) challenge.

Purpose
-------
Use historical query vectors from inputs_7.txt and historical scores from
outputs_6.txt to generate the next / 7th-round input query row.

Expected files in the same directory:
  - inputs_7.txt   : contains prior input rows. If it already contains row 7,
                     the script uses only the first len(outputs_6.txt) rows as
                     training history and can compare against the existing row 7.
  - outputs_6.txt  : contains objective outputs/scores for the first 6 rounds.

Output:
  - inputs_7_generated.txt : one line containing 8 numpy-array style query vectors

Assumption:
  - Objective is MAXIMIZATION. Change MAXIMIZE = False if your challenge is minimization.
  - All input parameters are normalized in [0, 1].
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


# -----------------------------
# Configuration
# -----------------------------
INPUTS_FILE = Path("inputs_7.txt")
OUTPUTS_FILE = Path("outputs_6.txt")
GENERATED_FILE = Path("inputs_7_generated.txt")

MAXIMIZE = True
RANDOM_SEED = 20260715
BOUNDS: Tuple[float, float] = (0.0, 1.0)
DECIMALS = 6

# With very few observed rounds, a large candidate pool gives BO a better chance
# to find useful Expected Improvement points.
N_GLOBAL_CANDIDATES = 80_000
N_LOCAL_CANDIDATES = 20_000
XI = 0.01  # larger = more exploration; smaller = more exploitation


# -----------------------------
# Robust parsing helpers
# -----------------------------
def read_logical_records(path: Path) -> List[str]:
    """Read records while handling numpy arrays that wrap across physical lines."""
    records: List[str] = []
    buffer: List[str] = []
    balance = 0

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        buffer.append(stripped)
        balance += stripped.count("[") - stripped.count("]")
        if balance == 0:
            records.append(" ".join(buffer))
            buffer = []

    if buffer:
        raise ValueError(f"Unbalanced brackets while parsing {path}: {' '.join(buffer)[:200]}")
    return records


def safe_eval_numpy_repr(text: str):
    """Safely evaluate strings containing array([...]) and np.float64(...)."""
    text = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
    return eval(text, {"__builtins__": {}}, {"array": np.array, "np": np})


def load_inputs(path: Path) -> List[List[np.ndarray]]:
    rows: List[List[np.ndarray]] = []
    for record in read_logical_records(path):
        rows.append([np.asarray(v, dtype=float) for v in safe_eval_numpy_repr(record)])
    return rows


def load_outputs(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for record in read_logical_records(path):
        rows.append([float(v) for v in safe_eval_numpy_repr(record)])
    return rows


# -----------------------------
# Bayesian optimization logic
# -----------------------------
def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float) -> np.ndarray:
    """Expected Improvement acquisition function."""
    sigma = np.maximum(sigma, 1e-12)
    if MAXIMIZE:
        improvement = mu - best_y - XI
    else:
        improvement = best_y - mu - XI
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def fit_gp_and_propose(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fit GP surrogate and propose the next query vector."""
    dim = X.shape[1]
    low, high = BOUNDS

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(dim) * 0.20,
                 length_scale_bounds=(1e-3, 10.0),
                 nu=2.5)
        + WhiteKernel(noise_level=1e-7, noise_level_bounds=(1e-12, 1e-2))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=20,
        alpha=1e-10,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, y)

    # Global candidates explore the complete domain.
    global_candidates = rng.uniform(low, high, size=(N_GLOBAL_CANDIDATES, dim))

    # Local candidates exploit around the currently best observed vector.
    best_idx = int(np.argmax(y) if MAXIMIZE else np.argmin(y))
    best_x = X[best_idx]
    local_candidates = best_x + rng.normal(loc=0.0, scale=0.075, size=(N_LOCAL_CANDIDATES, dim))
    local_candidates = np.clip(local_candidates, low, high)

    # Include small boundary probes; BBO optima sometimes sit near bounds.
    boundary_candidates = rng.uniform(low, high, size=(max(1000, 200 * dim), dim))
    mask = rng.random(boundary_candidates.shape) < 0.15
    boundary_candidates[mask] = rng.choice([low, high], size=mask.sum())

    candidates = np.vstack([global_candidates, local_candidates, boundary_candidates])
    mu, sigma = gp.predict(candidates, return_std=True)
    best_y = float(np.max(y) if MAXIMIZE else np.min(y))
    ei = expected_improvement(mu, sigma, best_y)

    # Choose the best non-duplicate candidate.
    for idx in np.argsort(ei)[::-1]:
        candidate = candidates[idx]
        min_dist = np.min(np.linalg.norm(X - candidate, axis=1))
        if min_dist > 1e-6:
            return np.round(candidate, DECIMALS)

    return np.round(rng.uniform(low, high, size=dim), DECIMALS)


def format_query_row(vectors: List[np.ndarray]) -> str:
    """Format as: [array([0.1, 0.2]), array([...]), ...]"""
    chunks = []
    for vector in vectors:
        values = ", ".join(f"{float(x):.{DECIMALS}f}" for x in vector)
        chunks.append(f"array([{values}])")
    return "[" + ", ".join(chunks) + "]\n"


def main() -> None:
    input_rows = load_inputs(INPUTS_FILE)
    output_rows = load_outputs(OUTPUTS_FILE)

    if not input_rows:
        raise ValueError("No input rows found.")
    if not output_rows:
        raise ValueError("No output rows found.")

    n_training_rows = len(output_rows)
    if len(input_rows) < n_training_rows:
        raise ValueError(
            f"inputs_7.txt has {len(input_rows)} rows but outputs_6.txt has {n_training_rows} rows."
        )

    training_inputs = input_rows[:n_training_rows]
    n_tasks = len(training_inputs[0])

    if any(len(row) != n_tasks for row in training_inputs):
        raise ValueError("Each input row must contain the same number of task vectors.")
    if any(len(row) != n_tasks for row in output_rows):
        raise ValueError("Each output row must contain one score per task vector.")

    rng = np.random.default_rng(RANDOM_SEED)
    proposed_vectors: List[np.ndarray] = []

    for task_idx in range(n_tasks):
        X = np.vstack([row[task_idx] for row in training_inputs])
        y = np.asarray([row[task_idx] for row in output_rows], dtype=float)
        proposed_vectors.append(fit_gp_and_propose(X, y, rng))

    GENERATED_FILE.write_text(format_query_row(proposed_vectors))

    print(f"Training rounds used: {n_training_rows}")
    print(f"Generated {n_tasks} query vectors and wrote: {GENERATED_FILE}")
    print(GENERATED_FILE.read_text())

    # If inputs_7.txt already contains a 7th row, show it so you can compare.
    if len(input_rows) > n_training_rows:
        existing = input_rows[n_training_rows]
        print("Existing next-row found in inputs_7.txt for comparison:")
        print(format_query_row(existing))


if __name__ == "__main__":
    main()
