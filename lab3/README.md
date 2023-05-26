# Communication-avoiding Krylov Methods

(Introductory text)

## What you will practice here

- evaluating numerical and computational efficiency of various algorithms
- selecting suitable solvers and preconditioners for a problem
- working with a large open source software package

## Your tasks

1. Try out different variants

- s-step GMRES
- GMRES + Polynomial preconditioner
- Gau\ss-Seidel with triangular solves replaced by Jacobi-Richardson (Polynomial variant of GS)

2. Performance model (roofline) analysis of these algorithms
3. Run on CPU and GPU, compare with model prediction
