# Lab 2: Iterative Methods in Trilinos

In this lab we will solve a PDE problem using data structures and solvers from the Trilinos framework.
We will assess the suitability of various solvers and preconditioners for CPU and GPU computing.


## What you will practice here

- compile and run MPI+X code (Modules, CMake, slurm, CPU and GPU nodes)
- deciding which algorithms are suitable for which operations
- deciding which algorithms are suitable for which applications

## Your tasks

- create a sparse matrix and fill it with the coefficients for a (non-symmetric) PDE problem
- solve it using various Krylov methods: look at timings for GMRES, GMRES(m) vs. short recurrence method like BiCGStab
- try out some preconditioning techniques, observe node-level performance of ILU vs. a polynomial method
- run the same experiments on CPU and GPU nodes with Kokkos/Ifpack2 etc.
- set up a one- and two-level Additive Schwarz preconditioner, experiment with different ways of splitting up the domain (CPU only)
