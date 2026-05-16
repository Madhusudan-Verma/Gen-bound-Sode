# Neural SDE Generalization Experiments

This repository contains experiments for studying the generalization behavior of Neural Stochastic Differential Equations (Neural SDEs). The implementation empirically evaluates how the generalization gap scales with:

- Sample size \(n\)
- Model dimension \(m\)
- Stochastic time horizon \(T\)

The experiments are implemented in PyTorch and trained on the MNIST dataset.

---

# Overview

Neural Stochastic Differential Equations (Neural SDEs) extend Neural Ordinary Differential Equations (Neural ODEs) by introducing stochastic dynamics into the hidden-state evolution process.

The hidden representation evolves according to:

\[
dH_t = f_\theta(H_t,t)\,dt + \epsilon\,dB_t,
\]

where:

- \(f_\theta\) is the learnable drift function,
- \(\epsilon\) controls stochastic noise,
- \(B_t\) is Brownian motion.

The experiments analyze how stochastic dynamics influence model complexity and generalization.

---

# Features

- Neural SDE implementation using Euler-Maruyama discretization
- Generalization gap analysis
- Scaling experiments with:
  - dataset size \(n\)
  - model dimension \(m\)
  - stochastic horizon \(T\)
- Multiple random seed averaging
- Automatic plotting of results
- MNIST dataset support

---

# Project Structure

```text
.
├── neural_sde_generalization_experiments.py
├── images/
│   ├── gen_gap_n.png
│   ├── gen_gap_m.png
│   └── gen_gap_T.png
├── references.bib
├── paper.tex
└── README.md
```

---

# Installation

Create a Python environment and install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

---

# How to Run

## 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

---

## 2. Create a Python Environment (Optional)

Using `venv`:

```bash
python -m venv nsde_env
```

Activate the environment:

### Linux / MacOS

```bash
source nsde_env/bin/activate
```

### Windows

```bash
nsde_env\Scripts\activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install torch torchvision matplotlib numpy
```

---

## 4. Run the Neural SDE Experiments

Execute the experiment script:

```bash
python neural_sde_generalization_experiments.py
```

---

# What the Script Does

The script automatically performs three experiments:

1. Generalization gap vs sample size \(n\)
2. Generalization gap vs model dimension \(m\)
3. Generalization gap vs stochastic time horizon \(T\)

For each experiment:
- the Neural SDE model is trained,
- the generalization gap is computed,
- results are averaged across multiple random seeds,
- plots are generated automatically.

---

# Neural SDE Model

The Neural SDE model is implemented as:

```python
dH_t = f_theta(H_t,t) dt + eps dB_t
```

using Euler-Maruyama discretization.

The drift function consists of multiple learnable neural blocks:

```python
f_theta(H_t) = sum(theta_i * f_i(H_t))
```

where each \(f_i\) is a small neural network.

---

# Experiments

## 1. Scaling with Sample Size \(n\)

Training set size is varied as:

```python
n = [100, 200, 500, 1000, 2000]
```

while keeping:

- \(m = 10\)
- \(T = 1.0\)

fixed.

Observation:
- Increasing \(n\) reduces the generalization gap.

---

## 2. Scaling with Model Dimension \(m\)

Hidden dimension is varied as:

```python
m = [2, 5, 10, 20, 40]
```

while fixing:

- \(n = 500\)
- \(T = 1.0\)

Observation:
- Larger \(m\) increases representational capacity but also increases generalization complexity.

---

## 3. Scaling with Time Horizon \(T\)

Time horizon is varied as:

```python
T = [0.5, 1.0, 1.5, 2.0, 3.0]
```

while fixing:

- \(m = 10\)
- \(n = 500\)

Observation:
- Larger stochastic horizons increase trajectory complexity and generalization gap.

---

# Generalization Gap

The generalization gap is computed as:

\[
\text{Gen Gap}
=
\mathcal{L}_{test}
-
\mathcal{L}_{train}
\]

using cross-entropy classification loss.

---

# Output

The generated plots are saved in the `images/` directory:

```text
images/
├── gen_gap_n.png
├── gen_gap_m.png
└── gen_gap_T.png
```

---

# Hardware Support

The implementation automatically detects GPU availability:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

If CUDA is available, training will run on GPU automatically.

---

# Expected Runtime

Approximate runtime on MNIST:

| Hardware | Runtime |
|---|---|
| CPU | 10–25 minutes |
| GPU | 2–8 minutes |

Runtime depends on:
- number of epochs,
- dataset size,
- model dimension,
- stochastic time horizon.

---

# Results

The experiments empirically validate the theoretical predictions:

- Generalization gap increases with:
  - stochastic time horizon \(T\)
  - model complexity \(m\)

- Generalization gap decreases with:
  - larger dataset size \(n\)

The observed scaling trends closely match theoretical Rademacher complexity bounds.

---

# Future Improvements

Potential future directions include:

- Adaptive Neural SDE solvers
- Learned discretization schemes
- Operator learning approaches
- Surrogate modeling for faster inference
- Stability regularization for stiff dynamics

---

# Troubleshooting

## CUDA Out of Memory

Reduce:
- batch size,
- hidden dimension,
- number of epochs.

---

## Missing Packages

Install dependencies manually:

```bash
pip install torch torchvision matplotlib numpy
```

---

## Slow Training

Try:
- reducing `steps`,
- reducing `epochs`,
- or using GPU acceleration.

---

# References

- Neural Ordinary Differential Equations
- Neural Stochastic Differential Equations
- Rademacher Complexity Theory
- Continuous-Time Deep Learning
- Stochastic Differential Equations

See `references.bib` for full citations.

---

# Author

Neural SDE Generalization Project
