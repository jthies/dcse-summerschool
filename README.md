# DCSE summerschool 2023 slides and exercise material

This repository contains slides and lab assignments for the summerschool,
it will be available for participants during the event.

## Overview of the lab sessions

### Lab 1, Monday morning: dense linear algebra and data dependencies

We will implement a dense LU decomposition using a block algorithm and OpenMP tasks  
[details](lab1/README.md)

### GPU lab, Monday afternoon: dense matrix-vector multiplication

We will implement General (dense) Matrix-Vector (GEMV) multiplication using OpenMP, 
and offload the calculation to the GPU. Different CUDA implementations highlight some
aspects of GPU hardware-aware programming.  
[details](gpu-lab/README.md)

### Lab 2, Tuesday: Sparse solvers and preconditioners in Trilinos

We will experiment with iterative solvers and preconditinoers from Trilinos, learn how to
comile and run applications for multi-core CPUs and GPUs, and investigate performance.  
[details](lab2/README.md)

## Lab 3, Wednesday: performance of SpMV in Trilinos

We will benchmark the Sparse Matrix-Vector (SpMV) multiplication in Trilinos for different matrices, a key operation for iterative linear and eigenvalue solvers.  
[details](lab3/README.md)

## Lab 4, Thursday: communication-avoiding Krylov methods

We will compare standard GMRES, GMRES with polynomial preconditioning and s-step GMRES methods in terms of convergence and performance.  
[details](lab4/README.md)

## Quantum Computing Lab, Friday: Introduction to Quantum Annealing

We will set up and solve a combinatorial optimiation problem on the D-Waves quantum annealer.
[details](qc-lab/README.md)

