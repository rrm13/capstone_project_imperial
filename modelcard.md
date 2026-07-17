# Model Card: Bayesian Optimization for Black-Box Optimization (BBO) Challenge

## Model Overview

This project uses Bayesian Optimization with a Gaussian Process (GP) surrogate model to generate candidate query vectors for an iterative Black-Box Optimization (BBO) challenge.

The workflow was used across multiple optimization rounds, where:

1. Historical input query vectors are collected.
2. Corresponding objective outputs are observed.
3. A Gaussian Process surrogate is fitted to the observed data.
4. An Expected Improvement (EI) acquisition function is used to propose the next query vectors.
5. The process repeats for subsequent rounds.

## Intended Use

### Primary Use

- Sequential black-box optimization.
- Expensive objective functions.
- Simulation optimization.
- Hyperparameter search.
- Challenge environments where only objective evaluations are available.

## Model Architecture

### Surrogate Model

Gaussian Process Regressor

Kernel:

- Constant Kernel
- Matern Kernel (nu=2.5)
- White Noise Kernel

### Acquisition Function

Expected Improvement (EI)

Balancing:

- Exploration of uncertain regions.
- Exploitation of high-performing regions.

### Search Strategy

The optimizer combines:

- Global random exploration.
- Local search around best-performing solutions.
- Boundary exploration near 0 and 1.
- Elite-solution perturbation.

## Inputs

Each optimization round contains:

- Multiple independent tasks.
- Query vectors normalized to the range [0,1].
- Variable-dimensional search spaces.

Examples observed in the challenge include:

- 2-dimensional vectors
- 3-dimensional vectors
- 4-dimensional vectors
- 5-dimensional vectors
- 6-dimensional vectors
- 8-dimensional vectors

## Outputs

The model produces:

- One candidate query vector per task.
- Values constrained to [0,1].
- Query proposals intended for the next optimization round.

## Assumptions

- The objective is treated as a maximization problem by default.
- Input parameters are scaled to [0,1].
- Observations may contain noise.
- Historical evaluations are representative of future behaviour.

## Training Data

This project does not use traditional training.

The GP surrogate is fitted dynamically using the historical optimization rounds available at runtime.

## Performance Characteristics

### Strengths

- Sample efficient.
- Effective for small datasets.
- Provides uncertainty estimates.
- Works well with expensive objective evaluations.

### Limitations

- GP scaling becomes expensive as observations grow.
- Performance may degrade in very high dimensions.
- Sensitive to noisy objectives.
- Can converge to local optima.

## Evaluation

Recommended evaluation metrics:

- Best objective value achieved.
- Improvement per round.
- Regret reduction.
- Convergence speed.
- Diversity of candidate solutions.

## Risks

Potential risks include:

- Premature convergence.
- Over-exploitation of historical optima.
- Poor surrogate fit for highly non-stationary objectives.
- Invalid recommendations if input normalization assumptions are violated.

## Reproducibility

Key configuration parameters:

- Fixed random seed.
- Gaussian Process surrogate.
- Matern kernel.
- Expected Improvement acquisition.
- Candidate pool sampling.

## Version Information

Project Name: BBO Bayesian Optimization Challenge

Model Type: Gaussian Process Bayesian Optimization

Acquisition Function: Expected Improvement

Status: Experimental / Research
