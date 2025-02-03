# Soap Films from Deep Learning: Neural Networks for Minimal Surfaces

## Introduction
Soap films naturally form shapes that minimize their surface area, known as **minimal surfaces**. Mathematically, finding these surfaces involves solving complex partial differential equations (PDEs). This project leverages deep learning to approximate minimal surfaces for arbitrary boundary conditions, bypassing traditional numerical methods. By training a neural network to parametrize the surface and minimize its area, we achieve accurate and efficient results.
![combined](https://github.com/user-attachments/assets/fb7b6536-836b-4c42-ac9c-5075fd566333)

## Approach
### Problem Formulation
A minimal surface can be parametrized by coordinates \( u \) and \( v \), with position vectors:
$$
\mathbf{r}(u, v) = \big(x(u,v),\, y(u,v),\, z(u,v)\big).
$$
The surface area is computed via the integral:
$$
\text{Area} = \iint_D \left\| \mathbf{r}_u \times \mathbf{r}_v \right\| \, du \, dv,
$$
where \( \mathbf{r}_u = \frac{\partial \mathbf{r}}{\partial u} \) and \( \mathbf{r}_v = \frac{\partial \mathbf{r}}{\partial v} \) are the partial derivatives. Minimizing this area while adhering to boundary constraints forms the core optimization problem.

### Neural Network as a Parametrization
- **Architecture**: A feedforward neural network with:
  - **Inputs**: \( u \) and \( v \) (domain parameters).
  - **Outputs**: \( x(u,v) \), \( y(u,v) \), \( z(u,v) \) (3D coordinates).
- **Loss Function**: Combines:
  1. **Area Loss**: Approximated via discrete integration over sampled \( (u,v) \) points:
     $$
     \mathcal{L}_{\text{area}} = \frac{1}{N} \sum_{i=1}^N \left\| \mathbf{r}_u^{(i)} \times \mathbf{r}_v^{(i)} \right\|
     $$
  2. **Boundary Loss**: Mean squared error (MSE) between predicted and true boundary points:
     $$
     \mathcal{L}_{\text{boundary}} = \frac{1}{M} \sum_{j=1}^M \left\| \mathbf{r}(u_j, v_j) - \mathbf{r}_{\text{target}}(u_j, v_j) \right\|^2
     $$

### Training
- **Optimizer**: Adam.
- **Epochs**: ~20,000.
- **Domain Considerations**: The domain of \( (u,v) \) must be topologically compatible with the target surface (e.g., an annulus for surfaces spanning two circles).

## Key Features
- **Topological Flexibility**: Choose domains (e.g., disks, annuli) that match the target surface's topology.
- **Efficiency**: Avoids solving PDEs directly, enabling faster computation for complex boundaries.
- **Physical Accuracy**: Accurately reproduces known minimal surfaces (e.g., catenoids, helicoids).
