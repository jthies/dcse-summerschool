# DCSE summerschool 2023 slides and exercise material

This repository contains slides and lab assignments for the summerschool,
it will be available for participants during the event.

## Overview of the lab sessions

### Lab 1, monday moring: dense linear algebra and data dependencies

We will implement a dense LU decomposition using OpenMP tasks  
[description](lab1/)
[description README](lab1/README)
[description README.md](lab1/README.md)

### GPU lab, monday afternoon: dense matrix-vector multiplication

We will implement General (dense) Matrix-Vector (GEMV) multiplication using OpenMP, 
and offload the calculation to the GPU. Different CUDA implementations highlight some
aspects of GPU hardware-aware programming.

### Lab 2, tuesday: Sparse solvers and preconditioners in Trilinos

We will experiment with iterative solvers and preconditinoers from Trilinos, learn how to
comile and run applications for multi-core CPUs and GPUs, and investigate performance.

## Lab 3, wednesday: performance of SpMV in Trilinos

We will benchmark the Sparse Matrix-Vector (SpMV) multiplication in Trilinos for different matrices, a key operation for iterative linear and eigenvalue solvers.

## Lab 4, thursday: communication-avoiding Krylov methods

We will compare standard GMRES, GMRES with polynomial preconditioning and s-step GMRES methods in terms of convergence and performance.

