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

$dH_t = f_\theta(H_t,t)dt +\epsilon dB_t$

where:

- $f_\theta$ is the learnable drift function.
- $\epsilon$ controls stochastic noise.
- $B_t$ is Brownian motion.

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
git clone https://github.com/Madhusudan-Verma/Gen-bound-Sode
cd Gen-bound-Sode
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


$dH_t = f_theta(H_t,t) dt + \epsilon dB_t$


using Euler-Maruyama discretization.

The drift function consists of multiple learnable neural blocks:


$f_{\theta}(H_t) = \sum(theta_i * f_i(H_t))$


where each $f_i$ is a small neural network.

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


$GenGap = L_{test} - L_{train}$


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

This project builds upon foundational work in continuous-depth deep learning, stochastic differential equations, and statistical learning theory. Key references include:

1. **Chen et al. (2018)** — *Neural Ordinary Differential Equations*  
   Introduced Neural ODEs as continuous-depth neural network architectures.

2. **Tzen and Raginsky (2019)** — *Theoretical Perspectives on Deep Generative Models Based on Stochastic Differential Equations*  
   Established theoretical connections between stochastic differential equations and deep generative learning.

3. **Kidger et al. (2021)** — *Neural Stochastic Differential Equations as Infinite-Dimensional GANs*  
   Developed practical Neural SDE training frameworks for stochastic sequence modeling.

4. **Øksendal (2003)** — *Stochastic Differential Equations: An Introduction with Applications*  
   Standard reference for stochastic calculus, Brownian motion, and SDE theory.

5. **Mohri, Rostamizadeh, and Talwalkar (2018)** — *Foundations of Machine Learning*  
   Reference for Rademacher complexity and statistical learning bounds.

6. **Wainwright (2019)** — *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*  
   Provides theoretical tools for complexity analysis and generalization theory.

7. **Ledoux and Talagrand (1991)** — *Probability in Banach Spaces*  
   Classical reference for entropy integrals and concentration inequalities.

For complete citation details, see:

```text
references.bib
```

---

# Author

Neural SDE Generalization Project

Research focus:
- Neural Stochastic Differential Equations (Neural SDEs)
- Generalization theory
- Continuous-time deep learning
- Stochastic dynamics in machine learning
- Complexity analysis and Rademacher bounds
