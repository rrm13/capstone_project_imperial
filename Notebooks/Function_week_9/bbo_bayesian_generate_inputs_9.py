#!/usr/bin/env python3
"""
Bayesian Optimization generator for BBO Challenge - Round 9 input queries.

This script uses:
  - inputs_9.txt   : historical input query rows. If this file already contains
                     a 9th row, the script uses only the first len(outputs_8.txt)
                     rows for training and prints the existing row for comparison.
  - outputs_8.txt  : objective values/scores for the first 8 rounds.

It writes:
  - inputs_9_generated.txt : one generated row of 8 query vectors for round 9.

Assumptions:
  - The objective is MAXIMIZATION. If this is a minimization challenge, set
    MAXIMIZE = False.
  - Input vector values are normalized in [0, 1].

Dependencies:
  pip install numpy scipy scikit-learn
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
INPUTS_FILE = Path("inputs_9.txt")
OUTPUTS_FILE = Path("outputs_8.txt")
GENERATED_FILE = Path("inputs_9_generated.txt")

MAXIMIZE = True
RANDOM_SEED = 20260715
BOUNDS: Tuple[float, float] = (0.0, 1.0)
DECIMALS = 6

# More candidates improves quality but increases runtime.
N_GLOBAL_CANDIDATES = 120_000
N_LOCAL_CANDIDATES = 40_000
N_BOUNDARY_CANDIDATES_PER_DIM = 350
XI = 0.01
MIN_DUPLICATE_DISTANCE = 1e-6


# -----------------------------
# File parsing helpers
# -----------------------------
def read_logical_records(path: Path) -> List[str]:
    """Read one logical list record at a time, handling wrapped numpy arrays."""
    records: List[str] = []
    buffer: List[str] = []
    bracket_balance = 0

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        buffer.append(stripped)
        bracket_balance += stripped.count("[") - stripped.count("]")

        if bracket_balance == 0 and buffer:
            records.append(" ".join(buffer))
            buffer = []

    if buffer:
        raise ValueError(f"Unbalanced brackets while parsing {path}: {' '.join(buffer)[:200]}")

    return records


def safe_eval_numpy_repr(text: str):
    """Evaluate strings containing array([...]) and np.float64(...)."""
    text = re.sub(r"np\.float64\(([^()]*)\)", r"\1", text)
    return eval(text, {"__builtins__": {}}, {"array": np.array, "np": np})


def load_inputs(path: Path) -> List[List[np.ndarray]]:
    rows: List[List[np.ndarray]] = []
    for record in read_logical_records(path):
        rows.append([np.asarray(vector, dtype=float) for vector in safe_eval_numpy_repr(record)])
    return rows


def load_outputs(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for record in read_logical_records(path):
        rows.append([float(value) for value in safe_eval_numpy_repr(record)])
    return rows


# -----------------------------
# Bayesian Optimization
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


def propose_for_task(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fit a GP surrogate for one task and propose the next vector."""
    dim = X.shape[1]
    low, high = BOUNDS

    # Matern kernel is a good default for black-box functions that may not be smooth.
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(dim) * 0.20,
            length_scale_bounds=(1e-3, 10.0),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-7, noise_level_bounds=(1e-12, 1e-2))
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        alpha=1e-10,
        n_restarts_optimizer=30,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)

    # 1) Global exploration across the whole [0, 1]^d domain.
    global_candidates = rng.uniform(low, high, size=(N_GLOBAL_CANDIDATES, dim))

    # 2) Local exploitation around the best observed point.
    best_idx = int(np.argmax(y) if MAXIMIZE else np.argmin(y))
    best_x = X[best_idx]
    local_candidates = best_x + rng.normal(0.0, 0.055, size=(N_LOCAL_CANDIDATES, dim))
    local_candidates = np.clip(local_candidates, low, high)

    # 3) Boundary-aware probes because some challenge functions peak near bounds.
    n_boundary = max(1000, N_BOUNDARY_CANDIDATES_PER_DIM * dim)
    boundary_candidates = rng.uniform(low, high, size=(n_boundary, dim))
    boundary_mask = rng.random(boundary_candidates.shape) < 0.20
    boundary_candidates[boundary_mask] = rng.choice([low, high], size=boundary_mask.sum())

    # 4) Small perturbed versions of top historical points to exploit known good regions.
    if MAXIMIZE:
        top_indices = np.argsort(y)[-min(3, len(y)):]
    else:
        top_indices = np.argsort(y)[:min(3, len(y))]
    elite_candidates = []
    for idx in top_indices:
        elite = X[idx] + rng.normal(0.0, 0.035, size=(max(2000, 500 * dim), dim))
        elite_candidates.append(np.clip(elite, low, high))
    elite_candidates = np.vstack(elite_candidates) if elite_candidates else np.empty((0, dim))

    candidates = np.vstack([global_candidates, local_candidates, boundary_candidates, elite_candidates])
    mu, sigma = model.predict(candidates, return_std=True)

    best_y = float(np.max(y) if MAXIMIZE else np.min(y))
    acquisition = expected_improvement(mu, sigma, best_y)

    # Select best non-duplicate candidate.
    for idx in np.argsort(acquisition)[::-1]:
        candidate = candidates[idx]
        distance_to_history = np.min(np.linalg.norm(X - candidate, axis=1))
        if distance_to_history > MIN_DUPLICATE_DISTANCE:
            return np.round(candidate, DECIMALS)

    # Very unlikely fallback.
    return np.round(rng.uniform(low, high, size=dim), DECIMALS)


def format_query_row(vectors: List[np.ndarray]) -> str:
    """Format as one BBO-compatible Python/numpy-style query row."""
    parts = []
    for vector in vectors:
        values = ", ".join(f"{float(value):.{DECIMALS}f}" for value in vector)
        parts.append(f"array([{values}])")
    return "[" + ", ".join(parts) + "]\n"


def main() -> None:
    input_rows = load_inputs(INPUTS_FILE)
    output_rows = load_outputs(OUTPUTS_FILE)

    n_training_rounds = len(output_rows)
    if len(input_rows) < n_training_rounds:
        raise ValueError(
            f"{INPUTS_FILE} contains {len(input_rows)} input rows but {OUTPUTS_FILE} "
            f"contains {n_training_rounds} output rows."
        )

    training_inputs = input_rows[:n_training_rounds]
    n_tasks = len(training_inputs[0])

    if any(len(row) != n_tasks for row in training_inputs):
        raise ValueError("All input rows used for training must have the same number of task vectors.")
    if any(len(row) != n_tasks for row in output_rows):
        raise ValueError("Every output row must contain one score per task vector.")

    rng = np.random.default_rng(RANDOM_SEED)
    generated_vectors: List[np.ndarray] = []

    for task_idx in range(n_tasks):
        X = np.vstack([row[task_idx] for row in training_inputs])
        y = np.asarray([row[task_idx] for row in output_rows], dtype=float)
        generated_vectors.append(propose_for_task(X, y, rng))

    GENERATED_FILE.write_text(format_query_row(generated_vectors))

    print(f"Training rounds used: {n_training_rounds}")
    print(f"Generated query vectors: {n_tasks}")
    print(f"Wrote: {GENERATED_FILE}")
    print(GENERATED_FILE.read_text())

    if len(input_rows) > n_training_rounds:
        print("Existing row in inputs_9.txt after the training rows, for comparison:")
        print(format_query_row(input_rows[n_training_rounds]))


if __name__ == "__main__":
    main()
