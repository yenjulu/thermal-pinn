# Heat Transfer PINN

A **Physics-Informed Neural Network (PINN)** for **2D transient heat conduction** on a unit square domain. The network learns the temperature field \(T(x, y, t)\) by minimizing a composite loss that enforces the governing PDE, boundary conditions, and initial condition.

---

## Table of Contents

1. [What is a PINN?](#what-is-a-pinn)
2. [Governing Physics](#governing-physics)
3. [Problem Setup](#problem-setup)
4. [Neural Network Approximation](#neural-network-approximation)
5. [Loss Function](#loss-function)
6. [Training Algorithm](#training-algorithm)
7. [Installation (Conda)](#installation-conda)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [References](#references)

---

## What is a PINN?

A **Physics-Informed Neural Network** embeds physical laws directly into the training objective. Instead of fitting data alone, the network is trained to:

1. **Satisfy a partial differential equation (PDE)** at collocation points in the domain interior.
2. **Satisfy boundary conditions (BC)** on the domain boundary.
3. **Satisfy initial conditions (IC)** at the starting time (for time-dependent problems).

The key idea is to approximate the solution \(u(\mathbf{x})\) with a neural network \(u_\theta(\mathbf{x})\) and use **automatic differentiation** to compute PDE derivatives. The total loss penalizes violations of physics and constraints:

\[
\mathcal{L}(\theta) = \mathcal{L}_{\text{PDE}} + \mathcal{L}_{\text{BC}} + \mathcal{L}_{\text{IC}}
\]

This approach was popularized by Raissi, Perdikaris, and Karniadakis (2019) for solving forward and inverse problems without labeled simulation data.

---

## Governing Physics

### Fourier's Law (1D)

Heat flux is proportional to the negative temperature gradient:

\[
q = -k \frac{\partial T}{\partial x}
\]

where \(k\) is thermal conductivity (W/m·K) and \(q\) is heat flux (W/m²).

### Conservation of Energy

For a homogeneous medium with no internal heat generation, energy conservation gives the **heat diffusion equation**:

\[
\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T)
\]

where:
- \(T\) — temperature (K or °C)
- \(t\) — time (s)
- \(\rho\) — density (kg/m³)
- \(c_p\) — specific heat capacity (J/kg·K)
- \(k\) — thermal conductivity (W/m·K)

### Thermal Diffusivity

Define the **thermal diffusivity**:

\[
\alpha = \frac{k}{\rho c_p} \quad \text{(m²/s)}
\]

For constant \(k\), the equation simplifies to:

\[
\frac{\partial T}{\partial t} = \alpha \nabla^2 T
\]

### 2D Form (This Project)

On a 2D domain \(\Omega \subset \mathbb{R}^2\):

\[
\boxed{
\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)
}
\]

This is a **parabolic PDE**: one time derivative and two spatial second derivatives.

### PDE Residual

For PINN training, define the **residual** \(f\) that should vanish when \(T\) is an exact solution:

\[
\boxed{
f(x, y, t) = \frac{\partial T}{\partial t} - \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right) = 0
}
\]

At each interior collocation point \((x_i, y_i, t_i)\), the network output must satisfy \(f = 0\).

---

## Problem Setup

| Quantity | Value / Form |
|----------|--------------|
| **Spatial domain** | \(\Omega = [0, 1] \times [0, 1]\) |
| **Time domain** | \(t \in [0, 1]\) |
| **Thermal diffusivity** | \(\alpha = 0.01\) (default, normalized units) |
| **Boundary condition** | Dirichlet: \(T = 0\) on \(\partial\Omega\) (cooled walls) |
| **Initial condition** | Gaussian hot spot at center at \(t = 0\): |

\[
T(x, y, 0) = \exp\left( -\frac{(x - x_0)^2 + (y - y_0)^2}{2\sigma^2} \right)
\]

Default: \(x_0 = y_0 = 0.5\), \(\sigma = 0.1\).

**Physical interpretation:** A localized heat pulse at the center of a plate diffuses outward and is absorbed by the cold boundary walls over time.

---

## Neural Network Approximation

The temperature field is approximated by a fully connected neural network (MLP):

\[
T_\theta(x, y, t) \approx T(x, y, t)
\]

where \(\theta\) denotes trainable weights and biases.

### Architecture

| Layer | Description |
|-------|-------------|
| Input | \((x, y, t) \in \mathbb{R}^3\) |
| Hidden | 4 layers × 64 neurons, activation = tanh |
| Output | \(T \in \mathbb{R}\) |

The tanh activation is standard in PINNs because it produces smooth outputs suitable for high-order derivative computation via autodiff.

### Derivative Computation

Spatial and temporal derivatives are computed using **automatic differentiation** (PyTorch `autograd`):

\[
\frac{\partial T_\theta}{\partial x}, \quad
\frac{\partial T_\theta}{\partial y}, \quad
\frac{\partial T_\theta}{\partial t}, \quad
\frac{\partial^2 T_\theta}{\partial x^2}, \quad
\frac{\partial^2 T_\theta}{\partial y^2}
\]

Second derivatives require nested autodiff (differentiate the first derivative with respect to the input coordinate).

---

## Loss Function

The total training loss is a weighted sum of three mean-squared-error terms:

\[
\boxed{
\mathcal{L}_{\text{total}} =
\lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} +
\lambda_{\text{BC}} \mathcal{L}_{\text{BC}} +
\lambda_{\text{IC}} \mathcal{L}_{\text{IC}}
}
\]

Default weights: \(\lambda_{\text{PDE}} = 1\), \(\lambda_{\text{BC}} = 10\), \(\lambda_{\text{IC}} = 10\).

### PDE Loss (Interior Collocation)

Sample \(N_{\text{int}}\) points \((x_i, y_i, t_i)\) uniformly in \(\Omega \times [0,1]\):

\[
\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left| f(x_i, y_i, t_i; \theta) \right|^2
\]

where:

\[
f = \frac{\partial T_\theta}{\partial t} - \alpha \left( \frac{\partial^2 T_\theta}{\partial x^2} + \frac{\partial^2 T_\theta}{\partial y^2} \right)
\]

### Boundary Condition Loss

Sample \(N_{\text{bc}}\) points on \(\partial\Omega\) at random times:

\[
\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{bc}}} \sum_{j=1}^{N_{\text{bc}}} \left| T_\theta(x_j, y_j, t_j) - T_{\text{bc}} \right|^2
\]

With \(T_{\text{bc}} = 0\) (Dirichlet).

### Initial Condition Loss

Sample \(N_{\text{ic}}\) points in \(\Omega\) at \(t = 0\):

\[
\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{ic}}} \sum_{k=1}^{N_{\text{ic}}} \left| T_\theta(x_k, y_k, 0) - T_{\text{ic}}(x_k, y_k) \right|^2
\]

where \(T_{\text{ic}}\) is the Gaussian hot-spot profile defined above.

---

## Training Algorithm

```
1. Initialize network T_θ with Xavier weights
2. Sample collocation points: interior, boundary, initial
3. For each epoch:
   a. Compute PDE residual f via autodiff → L_PDE
   b. Evaluate T_θ on boundary points       → L_BC
   c. Evaluate T_θ on initial points      → L_IC
   d. L_total = λ_PDE·L_PDE + λ_BC·L_BC + λ_IC·L_IC
   e. Backpropagate ∇_θ L_total
   f. Update θ with Adam optimizer
4. Save model and visualize T(x,y,t) snapshots
```

| Hyperparameter | Default |
|----------------|---------|
| Interior points | 4096 |
| Boundary points | 1024 |
| Initial points | 1024 |
| Epochs | 5000 |
| Learning rate | 1×10⁻³ |
| Optimizer | Adam |
| LR scheduler | StepLR (×0.5 every 2000 epochs) |

---

## Installation (Conda)

### Create the environment

From the project root:

```bash
conda env create -f environment.yml
conda activate thermal-pinn
```

If the environment already exists, update it:

```bash
conda env update -f environment.yml --prune
conda activate thermal-pinn
```

### Alternative: pip only

```bash
pip install -r requirements.txt
```

---

## Usage

Activate the conda environment, then run training:

```bash
conda activate thermal-pinn
python main.py
```

### Command-line options

```bash
python main.py --epochs 10000          # more training epochs
python main.py --alpha 0.05            # higher thermal diffusivity
python main.py --device cuda           # GPU acceleration
python main.py --lr 5e-4               # learning rate
python main.py --no-plot               # skip figure generation
python main.py --output-dir results    # custom output folder
```

### Outputs

After training, results are saved to `outputs/`:

| File | Description |
|------|-------------|
| `heat_pinn.pt` | Model checkpoint (weights + config) |
| `temperature_snapshots.png` | Temperature field at \(t = 0, 0.1, 0.25, 0.5, 1.0\) |
| `loss_history.png` | PDE, BC, IC, and total loss vs. epoch |

---

## Project Structure

```
thermal-pinn/
├── environment.yml       # Conda environment specification
├── requirements.txt      # Pip fallback dependencies
├── main.py               # CLI entry point
├── README.md
├── src/
│   ├── model.py          # HeatPINN neural network
│   ├── physics.py        # PDE residual, IC, collocation sampling
│   ├── train.py          # Training loop and configuration
│   └── visualize.py      # Temperature snapshots and loss plots
└── outputs/              # Generated after training
```

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
2. Özisik, M. N. (1993). *Heat Conduction*. Wiley.
3. Bergman, T. L., Lavine, A. S., Incropera, F. P., & DeWitt, D. P. (2011). *Fundamentals of Heat and Mass Transfer* (7th ed.). Wiley.
