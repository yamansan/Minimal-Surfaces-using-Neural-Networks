# Soap Films from Deep Learning: Neural Networks for Minimal Surfaces

## Introduction
Soap films naturally form shapes that minimize their surface area, known as **minimal surfaces**. Mathematically, finding these surfaces involves solving complex partial differential equations (PDEs). This project leverages deep learning to approximate minimal surfaces for arbitrary boundary conditions, bypassing traditional numerical methods. By training a neural network to parametrize the surface and minimize its area, we achieve accurate and efficient results.

## Approach
### Problem Formulation
A minimal surface can be parametrized by coordinates \( u \) and \( v \), with position vectors \( \mathbf{r}(u, v) = (x(u,v), y(u,v), z(u,v)) \). The surface area is computed via the integral:
\[
\text{Area} = \iint_D |r_u \times r_v| \, du \, dv
\]
where r_u and r_v are the partial derivatives of the position vector with respect to u and v. Minimizing this area while adhering to boundary constraints forms the core optimization problem.

### Neural Network as a Parametrization
- **Architecture**: A feedforward neural network with:
  - **Inputs**: \( u \) and \( v \) (domain parameters).
  - **Outputs**: \( x(u,v), y(u,v), z(u,v) \) (3D coordinates).
- **Loss Function**: Combines:
  1. **Area Loss**: Approximated via discrete integration over sampled \( (u,v) \) points.
  2. **Boundary Loss**: Mean squared error (MSE) between predicted and true boundary points.

### Training
- **Optimizer**: Adam.
- **Epochs**: ~20,000.
- **Domain Considerations**: The domain of \( (u,v) \) must be topologically compatible with the target surface (e.g., an annulus for surfaces spanning two circles).

## Key Features
- **Topological Flexibility**: Choose domains (e.g., disks, annuli) that match the target surface's topology.
- **Efficiency**: Avoids solving PDEs directly, enabling faster computation for complex boundaries.
- **Physical Accuracy**: Accurately reproduces known minimal surfaces (e.g., catenoids, helicoids).
