# Bayesian Optimization for Black-Box Optimization (BBO) Challenge

## Overview

This repository contains a Bayesian Optimization framework used to solve a multi-round Black-Box Optimization (BBO) challenge.

The approach uses:

- Gaussian Process (GP) surrogate models
- Matern kernels
- Expected Improvement (EI) acquisition function
- Adaptive exploration and exploitation
- Iterative optimization over multiple challenge rounds

The project generates candidate query vectors for the next optimization round based on historical inputs and observed objective values.

---

## Repository Structure

```text
.
├── inputs_*.txt
├── outputs_*.txt
├── bbo_bayesian_generate_inputs_*.py
├── modelcard.md
├── data_sheet.md
├── README.md
├── graphs/
│   ├── combined graphs
│   ├── symlog graphs
│   ├── 3D surface plots
│   └── 3D trajectory plots
```

---

## Methodology

### Surrogate Model

A Gaussian Process Regressor is trained using all previously observed rounds.

Kernel composition:

```python
ConstantKernel * Matern + WhiteKernel
```

### Acquisition Function

Expected Improvement (EI) is used to balance:

- Exploration of uncertain regions
- Exploitation of high-performing solutions

### Candidate Generation

The optimizer evaluates:

- Global random candidates
- Local perturbations near the current best solution
- Boundary candidates near 0 and 1
- Elite-solution perturbations

The candidate maximizing Expected Improvement is selected as the next query.

---

## Input Format

Example:

```python
[
    array([0.5, 0.5]),
    array([0.5, 0.5, 0.5]),
    ...
]
```

Each array represents one optimization task.

Observed tasks include:

- 2 dimensions
- 3 dimensions
- 4 dimensions
- 5 dimensions
- 6 dimensions
- 8 dimensions

---

## Output Format

Example:

```python
[
    array([0.12, 0.77]),
    array([0.32, 0.65, 0.41]),
    ...
]
```

Each generated vector is proposed for the next optimization round.

---

## Running the Optimizer

Example:

```bash
python bbo_bayesian_generate_inputs_13.py
```

Generated output:

```text
inputs_13_generated.txt
```

---

## Visualization

The project includes visualization utilities for:

### 2D Visualizations

- Combined output graphs
- Symlog-scaled objective graphs
- Convergence views

### 3D Visualizations

- 3D surface plots
- Signed-log surface plots
- Task trajectory plots

---

## Assumptions

- Objective is maximized.
- Decision variables are normalized to [0,1].
- Observations may contain noise.
- Historical samples are representative of future evaluations.

For minimization problems:

```python
MAXIMIZE = False
```

---

## Limitations

- GP models scale poorly with very large datasets.
- Performance may degrade in high-dimensional spaces.
- Results depend on the quality of historical observations.
- Black-box objectives may contain noise or discontinuities.

---

## Generated Documentation

- `modelcard.md` — model documentation
- `data_sheet.md` — dataset documentation

---

## Status

Research / Experimental

Designed for iterative Black-Box Optimization challenges and Bayesian Optimization benchmarking.
