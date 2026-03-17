# Physics-Informed Neural Networks (PINNs) for Fluid Dynamics

**Author:** Aadarsha Bhattarai  
**Category:** Scientific Machine Learning (SciML)  
**Problem:** Solving the 1D Burgers' Equation - Fluid Dynamics

---

## 1. Abstract
While traditional Neural Networks require massive datasets to learn patterns, **Physics-Informed Neural Networks (PINNs)** embed physical laws directly into the learning process. This project implements a PINN to solve the **1D Burgers' Equation**—a fundamental Partial Differential Equation (PDE) in fluid mechanics—demonstrating how AI can obey conservation laws even with minimal data.

## 2. The Physical Problem
The 1D Burgers' Equation is a simplified form of the Navier-Stokes equations, describing the evolution of a velocity field $u(x, t)$:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

* **Non-linear term ($u u_x$):** Causes the fluid to "pile up," leading to shock waves.
* **Diffusion term ($\nu u_{xx}$):** Represents viscosity, which acts to smooth the velocity field.

The challenge is the formation of a **shock wave** (a sharp discontinuity) at $x=0$ as time progresses.

## 3. Methodology & Architecture
I utilized **PyTorch** and **Automatic Differentiation** (`torch.autograd`) to build the solution.

* **Architecture:** 5-layer Multi-Layer Perceptron (MLP).
* **Activation:** **Tanh**, chosen for its infinite differentiability (required for second-order derivatives).
* **Composite Loss:** The model minimizes both data error and the physics residual:
    $$Loss = Loss_{InitialConditions} + Loss_{Physics}$$

## 4. Results
The model successfully captured the formation of the shock wave. Starting from a smooth sine wave at $t=0$, the network learned to maintain a sharp velocity transition as $t \to 1$, achieving a final loss of $\approx 1.4 \times 10^{-3}$.

## 5. Significance of the Results

### Mathematical Significance: Automatic Differentiation
By achieving a loss of $10^{-3}$, the model proves that the Neural Network has successfully approximated the solution to a non-linear PDE. Instead of using discrete numerical grids (like Finite Difference Methods), the model uses **Automatic Differentiation** (`torch.autograd`). This means the solution is continuous and differentiable across the entire space-time domain.

### Physics Significance: Shock Wave Capture
The resulting contour plot shows a crisp transition at $x=0$.
* **The Physics:** As time $t$ increases, the positive and negative velocity streams collide.
* **The Discovery:** The network "discovered" the shock wave without being shown any simulation data. It learned the shock purely by trying to minimize the physics residual $f$. This demonstrates that the model understands the balance between **nonlinear advection** and **viscous diffusion**.

## 6. Conclusion: AI for Science
This project serves as a proof-of-concept for **Scientific AI**. It shows that we can use AI not just to recognize patterns, but to solve complex physical systems. For my undergraduate studies, I aim to extend these principles to "AI for Science" applications like computational biophysics and drug discovery, where physical constraints are vital for accuracy.
